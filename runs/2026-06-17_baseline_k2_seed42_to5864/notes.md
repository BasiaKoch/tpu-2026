# baseline_k2_seed42_to5864

Rerun of the baseline hyperparameters on branch `baseline-k=2`, launched as a fresh W&B run.

Changed vs. baseline: `NUM_BATCHES` was increased from `3738` to `6516` so the existing derived formula gives `MAX_STEPS=5864` instead of `3364`. Everything else was held constant against the active baseline config defaults.

The reward function maps to baseline: `scripts/rewards.py` is byte-for-byte identical to `scripts/baseline_scripts/rewards.py`, with `REWARD_FNS = [match_format_exactly, match_format_approximately, check_answer, check_numbers]`.

## Startup

- Status: completed successfully.
- Start time: 2026-06-17T16:24:58Z.
- End time: 2026-06-17T23:56:42Z.
- Wall-clock time: 7h 31m 44s.
- Final step: 5864/5864.
- Exit code: 0.
- Final checkpoint: `/tmp/content/ckpts/actor/5864`.
- W&B run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/keh7f5es.
- Local W&B dir: `/home/funmilooi-somoye/tpu-2026/scripts/wandb/run-20260617_162508-keh7f5es`.
- Trainer printed: `Datasets: train=5864 val=652`.
- Trainer printed: `Starting GRPO training. CKPT_DIR=/tmp/content/ckpts/  MAX_STEPS=5864`.

## Caveats

- `~/.env` was loaded by the launch command. Observed relevant effective defaults: `SEED=42`, `BETA=0.08`, `DATA_SOURCE=tfds`, `TPU_CONTENT_DIR=/tmp/content`, `WANDB_PROJECT=tunix`, `WANDB_ENTITY=felsomoye-university-of-cambridge`.
- Current `train.py` does not set W&B group/job_type/tags, so those fields are not available from the run metadata.
- The local final checkpoint exists and was approximately 199M when checked.
- The final checkpoint was uploaded to W&B as
  `baseline_k2_seed42_to5864_checkpoint:v0`.
- TensorBoard logs were uploaded to W&B as
  `baseline_k2_seed42_to5864_tensorboard:v0`.
- Standalone baseline evaluation is still pending.

## Next Required Action

Run the baseline eval protocol and populate `eval_summary.txt`,
`base_eval.log`, and `lora_eval.log`.
