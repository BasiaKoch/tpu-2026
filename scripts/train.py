"""GRPO training entry point.

Run under tmux:
    tmux new -s tunix
    source ~/venvs/tunix/bin/activate
    cd ~/tpu-2026/scripts
    python train.py
    # detach: Ctrl-b d   # reattach: tmux attach -t tunix

Checkpoints are written to ~/checkpoints/<wandb_run_id>/ (one dir per run, so
runs never overwrite each other) and each committed step is uploaded to W&B as a
versioned model artifact during training.

Resuming a wandb run: set WANDB_RUN_ID in env (or pass --wandb-run-id). Because
the checkpoint dir is derived from the run id, resuming the same run id also
points Orbax at that run's existing checkpoints, and it picks up the latest step.
"""
import argparse
import glob
import os
import threading
import time
import types

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

import nest_asyncio
import optax
import wandb
from orbax import checkpoint as ocp
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

import config as train_config
from config import (
    B1, B2,
    BETA,
    DATA_SOURCE,
    EPSILON,
    EVAL_EVERY_N_STEPS,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MAX_PROMPT_LENGTH,
    MAX_STEPS,
    MAX_TO_KEEP,
    NUM_BATCHES,
    NUM_EPOCHS,
    NUM_GENERATIONS,
    NUM_ITERATIONS,
    NUM_TEST_BATCHES,
    SAVE_INTERVAL_STEPS,
    TEMPERATURE,
    TEST_DATA_DIR,
    TOP_K, TOP_P,
    TOTAL_GENERATION_STEPS,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MICRO_BATCH_SIZE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_RUN_ID,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from data import build_train_val_test
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer
from rewards import REWARD_FNS


def login_services():
    nest_asyncio.apply()  # tunix uses async; jupyter-style nesting helps in tmux too
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    if os.environ.get("HF_TOKEN"):
        os.system(f'hf auth login --token "{os.environ["HF_TOKEN"]}"')


def _config_as_dict() -> dict:
    return {
        k: v for k, v in vars(train_config).items()
        if not k.startswith("_") and not isinstance(v, types.ModuleType)
    }


def maybe_init_wandb(run_id: str | None):
    """Init wandb. If run_id is given we resume; otherwise a fresh run is created."""
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set — skipping wandb.")
        return None
    kwargs = {"project": WANDB_PROJECT, "entity": WANDB_ENTITY, "config": _config_as_dict()}
    if run_id:
        # "allow" => resume if the run exists on the server, otherwise create
        # a new run with this id. "must" errors out if the run was never synced
        # (which is what happens if the previous training crashed before
        # wandb.init was reached).
        kwargs.update({"id": run_id, "resume": "allow"})
    run = wandb.init(**kwargs)
    config_path = os.path.abspath(train_config.__file__)
    wandb.save(config_path, base_path=os.path.dirname(config_path), policy="now")
    print(f"Uploaded W&B config snapshot: {config_path}")
    return run


def build_optimizer():
    schedule = optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    )
    opt = optax.adamw(learning_rate=schedule, b1=B1, b2=B2, weight_decay=WEIGHT_DECAY)
    if MAX_GRAD_NORM is not None:
        opt = optax.chain(optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM), opt)
    return opt


def build_cluster_config(mesh, optimizer, eos_tokens, ckpt_dir, tensorboard_dir):
    return rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=EVAL_EVERY_N_STEPS,
            max_steps=MAX_STEPS,
            mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
            train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
            metrics_logging_options=metrics_logger.MetricsLoggerOptions(
                log_dir=tensorboard_dir, flush_every_n_steps=20,
            ),
            checkpoint_root_directory=ckpt_dir,
            checkpointing_options=ocp.CheckpointManagerOptions(
                save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP,
            ),
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=TOTAL_GENERATION_STEPS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            eos_tokens=eos_tokens,
        ),
    )


def start_checkpoint_uploader(run, ckpt_dir, stop_event, poll_seconds=120, settle_seconds=30):
    """Log each committed Orbax checkpoint to W&B as it appears during training.

    tunix has no per-save hook, so we watch the filesystem instead. It writes
    actor checkpoints under <ckpt_dir>/actor/<step>/; a step is finalised once
    Orbax has written its `_CHECKPOINT_METADATA` marker. We treat a step as ready
    only when that marker has been stable for `settle_seconds`, so we never upload
    a checkpoint that is still being flushed. Each step is logged as a new version
    of a single 'model' artifact, aliased `step-<n>`. Upload errors are caught so
    they can never take down the training run.

    Returns the started daemon thread; call stop_event.set() then join() it after
    training so the final checkpoint(s) get a last upload pass.
    """
    actor_dir = os.path.join(ckpt_dir, "actor")
    artifact_name = f"{run.id}-actor-ckpt"
    uploaded: set[str] = set()

    def ready_steps(min_age):
        steps = {}
        for marker in glob.glob(os.path.join(actor_dir, "*", "_CHECKPOINT_METADATA")):
            step_dir = os.path.dirname(marker)
            step = os.path.basename(step_dir)
            if step.isdigit() and (time.time() - os.path.getmtime(marker)) >= min_age:
                steps[step] = step_dir
        return dict(sorted(steps.items(), key=lambda kv: int(kv[0])))

    def sweep(min_age):
        for step, step_dir in ready_steps(min_age).items():
            if step in uploaded:
                continue
            try:
                art = wandb.Artifact(artifact_name, type="model", metadata={"step": int(step)})
                art.add_dir(step_dir)
                run.log_artifact(art, aliases=[f"step-{step}"])
                uploaded.add(step)
                print(f"[ckpt-uploader] logged checkpoint step {step} to W&B")
            except Exception as e:  # never let an upload error kill training
                print(f"[ckpt-uploader] failed to upload step {step}: {e}")

    def loop():
        while not stop_event.is_set():
            sweep(settle_seconds)
            stop_event.wait(poll_seconds)
        sweep(min_age=0)  # final pass: catch the last checkpoint(s) after training ends

    t = threading.Thread(target=loop, name="ckpt-uploader", daemon=True)
    t.start()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DATA_SOURCE, choices=["tfds", "kaggle"])
    ap.add_argument("--wandb-run-id", default=WANDB_RUN_ID,
                    help="Pass an existing run id (e.g. bnh9ttlt) to resume.")
    args = ap.parse_args()

    login_services()
    # init wandb BEFORE the trainer because tunix sometimes hangs if wandb is
    # initialised mid-RLCluster construction (known bug).
    run = maybe_init_wandb(args.wandb_run_id)

    # Save each run's checkpoints under ~/checkpoints/<wandb_run_id> so a new run
    # never overwrites or evicts a previous run's checkpoints (the run id isn't
    # known until after wandb.init, which is why this isn't baked into config.py).
    # Falls back to a timestamped dir when wandb is disabled.
    run_id = run.id if run is not None else time.strftime("local-%Y%m%d-%H%M%S")
    ckpt_dir = os.path.expanduser(os.path.join("~/checkpoints", run_id))
    tensorboard_dir = os.path.join(ckpt_dir, "tensorboard", "grpo")
    print(f"Checkpoints -> {ckpt_dir}")

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, _ = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    train_ds, val_ds, _ = build_train_val_test(
        NUM_BATCHES, NUM_TEST_BATCHES, TRAIN_MICRO_BATCH_SIZE, TRAIN_FRACTION,
        NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR, source=args.source,
    )
    print(f"Datasets: train={len(train_ds)} val={len(val_ds) if val_ds else 0}")

    optimizer = build_optimizer()
    cluster_cfg = build_cluster_config(mesh, optimizer, eos_tokens, ckpt_dir, tensorboard_dir)
    grpo_cfg = GRPOConfig(
        num_generations=NUM_GENERATIONS,
        num_iterations=NUM_ITERATIONS,
        beta=BETA,
        epsilon=EPSILON,
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=lora, reference=base, tokenizer=tokenizer, cluster_config=cluster_cfg,
    )
    trainer = GRPOLearner(rl_cluster=rl_cluster, reward_fns=REWARD_FNS, algo_config=grpo_cfg)

    print(f"Starting GRPO training. CKPT_DIR={ckpt_dir}  MAX_STEPS={MAX_STEPS}")

    # Upload intermediate checkpoints to W&B as they are written during training.
    stop_uploading = threading.Event()
    uploader = start_checkpoint_uploader(run, ckpt_dir, stop_uploading) if run is not None else None
    try:
        trainer.train(train_ds, val_ds)
    finally:
        stop_uploading.set()
        if uploader is not None:
            uploader.join(timeout=900)  # let the final-sweep upload finish
    print("Training finished.")


if __name__ == "__main__":
    main()
