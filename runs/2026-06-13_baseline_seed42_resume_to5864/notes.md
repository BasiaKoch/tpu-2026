# Baseline seed 42 continuation to step 5864

Date: 2026-06-13

Purpose: continue the completed baseline run from global step 3364 to global step 5864.

## Step target

- Previous completed step: `3364`
- Target total step: `5864`
- Additional training steps required: `2500`
- Trainer `MAX_STEPS` should be set to the absolute global target: `5864`

## Restored checkpoint

The documented artifact name `baseline_seed42_checkpoint:latest` was not found in W&B under `felsomoye-university-of-cambridge/tunix`.

Available artifact used instead:

- W&B artifact: `felsomoye-university-of-cambridge/tunix/Checkpoint-baseline-k-2:v0`
- Artifact type shown by W&B: `model`
- Downloaded artifact directory: `artifacts/Checkpoint-baseline-k-2_v0`
- Downloaded artifact size: `233M`

The artifact contained a single Orbax checkpoint payload directly at the artifact root:

- `_CHECKPOINT_METADATA`
- `model_params/`
- `optimizer_state/`

It was copied into the Tunix resume layout:

- Resume checkpoint root: `/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864`
- Actor step checkpoint: `/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864/actor/3364`

Verification after copy:

- `actor/3364` step directory exists
- `_CHECKPOINT_METADATA` exists
- `model_params/` exists
- `optimizer_state/` exists
- File count under `actor/3364`: `33`

## `/tmp` status before launch

Checked before restoring/launching:

- `/tmp/content` did not exist
- `/tmp/content/ckpts` did not exist
- `/tmp/content/intermediate_ckpt` did not exist
- `/tmp/content/tmp/tensorboard/grpo` did not exist
- `/tmp` had about `79G` available

Conclusion: `/tmp` was clean enough for a fresh training launch.

## Required config for continuation

Use a persistent checkpoint root rather than `/tmp`:

```python
NUM_EPOCHS = 2
MAX_STEPS = 5864
CKPT_DIR = "/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 8
```

`NUM_EPOCHS=2` is needed because the original default dataset length provides about 3364 train batches. The continuation needs enough available batches for the trainer to reach global step 5864.

## W&B resume details

Previous completed baseline run:

- W&B run URL: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl`
- W&B run id: `jgs4c6kl`
- W&B project: `tunix`
- W&B entity: `felsomoye-university-of-cambridge`

Suggested launch command:

```bash
cd /home/funmilooi-somoye/tpu-2026/scripts
WANDB_ENTITY=felsomoye-university-of-cambridge \
WANDB_PROJECT=tunix \
WANDB_RUN_ID=jgs4c6kl \
python -u train.py --wandb-run-id jgs4c6kl
```

## Checks still needed at launch

- Confirm startup logs indicate checkpoint restore from step `3364`.
- Confirm training continues past step `3364`, not from step `0`. Done: TensorBoard actor train metrics start at step `3365` and were observed through step `3391`.
- Confirm final checkpoint reaches step `5864`.
- Preserve the final checkpoint outside `/tmp` if any config still writes temporary artifacts there.

## Launch status

Launched on 2026-06-13 in tmux session `tunix_resume_5864`.

Two venv fixes were needed before successful launch:

- Installed missing `nest-asyncio==1.6.0`
- Installed missing `libtpu==0.0.40`

After installing `libtpu`, JAX outside the sandbox reported one TPU device:

```text
[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

Resume was confirmed from TensorBoard event data:

- `actor/train/loss` first observed step: `3365`
- `actor/train/loss` latest observed step at check time: `3391`

This confirms continuation from the restored step-`3364` checkpoint.
