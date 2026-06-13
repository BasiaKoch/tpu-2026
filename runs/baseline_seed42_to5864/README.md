# Baseline seed 42 continuation to step 5864

Canonical run metadata folder for the continuation of `baseline_seed42` from
step `3364` to target global step `5864`.

## Current status

- Status: running
- tmux session: `tunix_resume_5864`
- training process: `python -u train.py --wandb-run-id jgs4c6kl`
- observed training PID: `33810`
- latest saved checkpoint observed: `3500`
- latest W&B/current actor step observed from tmux output: `3619`
- target final step: `5864`

## Live files

The original dated run record remains the live log location:

- Dated run folder: `/home/funmilooi-somoye/tpu-2026/runs/2026-06-13_baseline_seed42_resume_to5864`
- Live train log: `/home/funmilooi-somoye/tpu-2026/runs/2026-06-13_baseline_seed42_resume_to5864/train.log`
- TensorBoard event file: `/tmp/content/tmp/tensorboard/grpo/events.out.tfevents.1781373532.t1v-n-e9b5fb70-w-0`

Attach/monitor:

```bash
tmux attach -t tunix_resume_5864
tail -f /home/funmilooi-somoye/tpu-2026/runs/2026-06-13_baseline_seed42_resume_to5864/train.log
```

## Resume confirmation

The run restored from checkpoint step `3364` and continued with the next actor
optimizer step `3365`.

Evidence:

- restored checkpoint root: `/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864`
- pre-launch checkpoint: `actor/3364`
- TensorBoard actor metrics first observed step: `3365`
- checkpoint `actor/3500` was saved after launch

Note: W&B shows warnings for reward/rollout metrics logged at local steps
starting from `0` while the resumed W&B run's current step is already in the
global actor-step range. Actor training metrics are the reliable continuation
signal here.

## W&B

- Entity: `felsomoye-university-of-cambridge`
- Project: `tunix`
- Run id: `jgs4c6kl`
- Run URL: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl`
- Run name shown by W&B: `Baseline k=2`

## Source checkpoint

Documented artifact `baseline_seed42_checkpoint:latest` was not present in
W&B. The checkpoint restored for this continuation came from:

- W&B artifact: `felsomoye-university-of-cambridge/tunix/Checkpoint-baseline-k-2:v0`
- local artifact download: `/home/funmilooi-somoye/tpu-2026/artifacts/Checkpoint-baseline-k-2_v0`
- staged checkpoint: `/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864/actor/3364`

## Run configuration

The continuation uses the same baseline hyperparameters except for:

- `NUM_EPOCHS = 2`
- `MAX_STEPS = 5864`
- `CKPT_DIR = "/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864"`
- `MAX_TO_KEEP = 8`

See `config_metadata.txt` for the full captured config values.
