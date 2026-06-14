# Baseline seed 42 continuation to step 5864

Canonical run metadata folder for the continuation of `baseline_seed42` from
step `3364` to target global step `5864`.

## Final status

- Status: completed successfully
- Launch time: `2026-06-13 17:58 UTC`
- End time: `2026-06-13 18:37:59 UTC`
- Exit code: `0`
- tmux session: `tunix_resume_5864` remains open at a shell after process exit
- training process: finished; no active `python -u train.py` process remains
- restored checkpoint: `actor/3364`
- final step: `5864`
- final checkpoint: `/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864/actor/5864`
- final checkpoint size: approximately `200M`
- checkpoint root size: approximately `1.3G`

## Run files

The original dated run record remains the tee log location:

- Dated run folder: `/home/funmilooi-somoye/tpu-2026/runs/2026-06-13_baseline_seed42_resume_to5864`
- Train log: `/home/funmilooi-somoye/tpu-2026/runs/2026-06-13_baseline_seed42_resume_to5864/train.log`
- Train log size: approximately `19M`
- TensorBoard event file: `/tmp/content/tmp/tensorboard/grpo/events.out.tfevents.1781373532.t1v-n-e9b5fb70-w-0`
- W&B local run directory: `/home/funmilooi-somoye/tpu-2026/scripts/wandb/run-20260613_175840-jgs4c6kl`
- W&B local run directory size: approximately `61M`

## Resume confirmation

The run restored from checkpoint step `3364` and continued with the next actor
optimizer step `3365`.

Evidence:

- restored checkpoint root: `/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864`
- pre-launch checkpoint: `actor/3364`
- training startup log: `CKPT_DIR=/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864  MAX_STEPS=5864`
- TensorBoard actor metrics first observed step: `3365`
- intermediate checkpoints saved: `3500`, `4000`, `4500`, `5000`, `5500`
- final checkpoint saved: `5864`
- W&B summary final `_step`: `5864`

Note: W&B shows warnings for reward/rollout metrics logged at local steps
starting from `0` while the resumed W&B run's current step is already in the
global actor-step range. Actor training metrics, checkpoint saves, and W&B
summary step confirm the actual continuation.

## W&B

- Entity: `felsomoye-university-of-cambridge`
- Project: `tunix`
- Run id: `jgs4c6kl`
- Run URL: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl`
- Run name shown by W&B: `Baseline k=2`
- Local W&B summary file: `/home/funmilooi-somoye/tpu-2026/scripts/wandb/run-20260613_175840-jgs4c6kl/files/wandb-summary.json`
- Local W&B final sync evidence: `filestream` request completed with `status: 200 OK`, `complete: true`, `exit_code: 0`
- Final checkpoint artifact: `felsomoye-university-of-cambridge/tunix/baseline_seed42_to5864_checkpoint:v0`
- Final checkpoint artifact URL: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/artifacts/model/baseline_seed42_to5864_checkpoint/v0`
- Final checkpoint artifact aliases: `latest`, `step-5864`

The continuation was linked to the existing W&B run `jgs4c6kl`; it did not
create a separate W&B run for the 5864 continuation.

## Source checkpoint

Documented artifact `baseline_seed42_checkpoint:latest` was not present in
W&B. The checkpoint restored for this continuation came from:

- W&B artifact: `felsomoye-university-of-cambridge/tunix/Checkpoint-baseline-k-2:v0`
- local artifact download: `/home/funmilooi-somoye/tpu-2026/artifacts/Checkpoint-baseline-k-2_v0`
- staged checkpoint: `/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864/actor/3364`

## Uploaded final checkpoint

The final `actor/5864` checkpoint was uploaded to W&B on 2026-06-13:

- Artifact: `felsomoye-university-of-cambridge/tunix/baseline_seed42_to5864_checkpoint:v0`
- URL: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/artifacts/model/baseline_seed42_to5864_checkpoint/v0`
- Aliases: `latest`, `step-5864`
- Type: `model`
- Attached run: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl`
- Upload source: `/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864/actor/5864`
- CLI verification: `wandb artifact get felsomoye-university-of-cambridge/tunix/baseline_seed42_to5864_checkpoint:v0` resolved and downloaded the artifact.

## Run configuration

The continuation uses the same baseline hyperparameters except for:

- `NUM_EPOCHS = 2`
- `MAX_STEPS = 5864`
- `CKPT_DIR = "/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864"`
- `MAX_TO_KEEP = 8`

See `config_metadata.txt` for the full captured config values.
