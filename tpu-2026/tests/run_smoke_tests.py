#!/usr/bin/env python3
"""TPU smoke tests for the Gemma/Tunix GRPO repo.

These checks are intentionally tiny. They validate environment, data, model
loading, one training step, local logging/checkpointing, and one evaluation
example without modifying the repo's training config files.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_ROOT = Path("/tmp/content/debug_smoke_test_do_not_report")

sys.path.insert(0, str(SCRIPTS_DIR))


def print_header(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def load_local_env() -> None:
    load_dotenv(Path.home() / ".env")


def smoke_env(require_wandb: bool = False) -> None:
    print_header("env")
    import grain  # noqa: F401
    import jax
    import tensorflow_datasets as tfds  # noqa: F401
    import tunix  # noqa: F401
    import flax  # noqa: F401

    print("python", sys.version.split()[0])
    print("platform", platform.platform())
    print("jax_backend", jax.default_backend())
    print("jax_devices", jax.devices())

    required = ["HF_TOKEN"]
    optional = ["WANDB_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY"]
    for name in required + optional:
        value = os.getenv(name)
        print(f"{name}_set", bool(value))
        if name in required and not value:
            raise RuntimeError(f"{name} is required for smoke tests but is not set")

    wandb_key = os.getenv("WANDB_API_KEY")
    if require_wandb and (not wandb_key or len(wandb_key) < 40):
        raise RuntimeError("WANDB_API_KEY is required and must be a valid W&B key")
    if wandb_key and len(wandb_key) < 40:
        print("WANDB_API_KEY_warning invalid length; W&B is not validated")


def smoke_dataset(root: Path, source: str) -> None:
    print_header("dataset")
    from data import build_train_val_test

    run_root = root / f"dataset_{int(time.time())}"
    train_ds, val_ds, test_ds = build_train_val_test(
        num_batches=1,
        num_test_batches=1,
        train_micro_batch_size=1,
        train_fraction=1.0,
        num_epochs=1,
        train_dir=str(run_root / "data" / "train"),
        test_dir=str(run_root / "data" / "test"),
        source=source,
    )
    first = next(iter(train_ds))
    print("root", run_root)
    print("lengths", {"train": len(train_ds), "val": 0 if val_ds is None else len(val_ds), "test": len(test_ds)})
    print("keys", sorted(first.keys()))
    print("sample_answer", first["answer"][0])


def smoke_model() -> None:
    print_header("model")
    from model import build_mesh, download_weights, get_lora_model, load_base_model, load_tokenizer

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    _ = get_lora_model(base, mesh)
    _, eos_tokens = load_tokenizer(eos_tokens)
    print("local_path", local_path)
    print("num_layers", cfg.num_layers)
    print("eos_tokens", eos_tokens)


def force_tensorboard_only_metrics(train_module, log_dir: str):
    from tunix.sft import metrics_logger as ml

    original_options = train_module.metrics_logger.MetricsLoggerOptions

    def tensorboard_only_options(log_dir, flush_every_n_steps=100, **kwargs):
        return original_options(
            log_dir=log_dir,
            flush_every_n_steps=flush_every_n_steps,
            backend_kwargs={
                "custom_backend": [
                    lambda: ml.TensorboardBackend(
                        log_dir=log_dir,
                        flush_every_n_steps=flush_every_n_steps,
                    )
                ]
            },
        )

    train_module.metrics_logger.MetricsLoggerOptions = tensorboard_only_options


def smoke_train(root: Path, source: str) -> Path:
    print_header("train")
    import train

    os.environ.pop("WANDB_API_KEY", None)
    os.environ["WANDB_MODE"] = "disabled"

    run_root = root / f"train_{int(time.time())}"
    train.NUM_BATCHES = 1
    train.NUM_TEST_BATCHES = 1
    train.TRAIN_FRACTION = 1.0
    train.NUM_EPOCHS = 1
    train.MAX_STEPS = 1
    train.WARMUP_STEPS = 0
    train.EVAL_EVERY_N_STEPS = 1
    train.SAVE_INTERVAL_STEPS = 1
    train.MAX_TO_KEEP = 1
    train.TRAIN_DATA_DIR = str(run_root / "data" / "train")
    train.TEST_DATA_DIR = str(run_root / "data" / "test")
    train.CKPT_DIR = str(run_root / "ckpts")
    train.INTERMEDIATE_CKPT_DIR = str(run_root / "intermediate_ckpt")
    train.TENSORBOARD_DIR = str(run_root / "tensorboard" / "grpo")
    train.login_services = lambda: print("external service login skipped for local smoke")
    train.maybe_init_wandb = lambda run_id: (print("wandb init skipped for local smoke"), None)[1]
    force_tensorboard_only_metrics(train, train.TENSORBOARD_DIR)

    old_argv = sys.argv[:]
    try:
        sys.argv = ["train.py", "--source", source]
        print("root", run_root)
        print("max_steps", train.MAX_STEPS)
        print("checkpoint_dir", train.CKPT_DIR)
        print("tensorboard_dir", train.TENSORBOARD_DIR)
        train.main()
    finally:
        sys.argv = old_argv

    event_files = list((run_root / "tensorboard" / "grpo").glob("events.out.tfevents.*"))
    ckpt_dir = run_root / "ckpts" / "actor" / "1"
    if not event_files:
        raise RuntimeError("training smoke finished but no TensorBoard event file was found")
    if not ckpt_dir.exists():
        raise RuntimeError("training smoke finished but actor checkpoint step 1 was not found")
    print("tensorboard_event", event_files[0])
    print("actor_checkpoint", ckpt_dir)
    return run_root


def smoke_eval(root: Path, source: str) -> None:
    print_header("eval")
    import evaluate as ev
    from tunix.generate import sampler as sampler_lib

    run_root = root / f"eval_{int(time.time())}"
    ev.NUM_BATCHES = 1
    ev.NUM_TEST_BATCHES = 1
    ev.TRAIN_MICRO_BATCH_SIZE = 1
    ev.TRAIN_FRACTION = 1.0
    ev.NUM_EPOCHS = 1
    ev.TRAIN_DATA_DIR = str(run_root / "data" / "train")
    ev.TEST_DATA_DIR = str(run_root / "data" / "test")
    ev.TOTAL_GENERATION_STEPS = 64

    mesh = ev.build_mesh()
    local_path, eos_tokens = ev.download_weights()
    base, cfg = ev.load_base_model(local_path, mesh)
    lora = ev.get_lora_model(base, mesh)
    tokenizer, eos_tokens = ev.load_tokenizer(eos_tokens)
    _, _, test_ds = ev.build_train_val_test(
        ev.NUM_BATCHES,
        ev.NUM_TEST_BATCHES,
        ev.TRAIN_MICRO_BATCH_SIZE,
        ev.TRAIN_FRACTION,
        ev.NUM_EPOCHS,
        ev.TRAIN_DATA_DIR,
        ev.TEST_DATA_DIR,
        source=source,
    )
    sampler = sampler_lib.Sampler(
        transformer=lora,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=ev.MAX_PROMPT_LENGTH + ev.TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )
    print("root", run_root)
    n, total, acc, partial_acc, format_acc = ev.evaluate(
        test_ds,
        sampler,
        eos_tokens,
        temperature=None,
        top_k=1,
        top_p=None,
        num_passes=1,
    )
    print(
        "eval_smoke_final",
        {
            "correct": f"{n}/{total}",
            "acc": round(acc, 2),
            "partial": round(partial_acc, 2),
            "format": round(format_acc, 2),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny TPU smoke tests.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "env", "dataset", "model", "train", "eval"],
        help="Smoke stages to run.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", default="tfds", choices=["tfds", "kaggle"])
    parser.add_argument("--require-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env()
    args.root.mkdir(parents=True, exist_ok=True)

    stages = args.stages
    if "all" in stages:
        stages = ["env", "dataset", "model", "train", "eval"]

    if "env" in stages:
        smoke_env(require_wandb=args.require_wandb)
    if "dataset" in stages:
        smoke_dataset(args.root, args.source)
    if "model" in stages:
        smoke_model()
    if "train" in stages:
        smoke_train(args.root, args.source)
    if "eval" in stages:
        smoke_eval(args.root, args.source)

    print("\nSmoke tests completed.")


if __name__ == "__main__":
    main()
