# baseline_seed42

Patched official baseline run.

The earlier baseline attempts from this date were moved to `runs/2026-06-08_baseline_seed42_failed/`. Those attempts failed before training during TFDS GSM8K dataset construction.

This run uses commit `7e696c428687bc083604690f5f51009da6abb6d9`, which pins `protobuf==6.31.1` to fix the TFDS/protobuf failure. No model, dataset, reward, or training hyperparameter changes are intended beyond that dependency/runtime patch.

Live stdout/stderr was written to `scripts/train.log`. The tmux launcher remained open after training by design, but the `python -u train.py` process exited successfully.

## Outcome

- Status: completed and evaluated.
- Start time: 2026-06-08T18:17:09Z.
- End time: 2026-06-08T22:58:47Z.
- Wall-clock: 4h 41m 38s.
- Final step: 3364.
- Exit code: 0.
- Final checkpoint: `/tmp/content/ckpts/actor/3364`.
- Preserved local checkpoint copy: `/home/ext_felsomoye_gmail_com/tpu-2026/checkpoints/baseline_seed42/actor/3364`.
- W&B run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl.
- W&B checkpoint artifact: `felsomoye-university-of-cambridge/tunix/baseline_seed42_checkpoint:latest`.
- W&B artifact upload run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/bah3rczo.
- Final log marker: `Training finished.`

Evaluation completed with greedy decoding on the configured 64-example GSM8K test eval. Base model / zero-init LoRA wrapper: `33/64`, `51.56%` accuracy, `53.12%` partial, `6.25%` format. Restored LoRA checkpoint step `3364`: `2/64`, `3.12%` accuracy, `6.25%` partial, `12.50%` format. Logs are `base_eval.log` and `lora_eval.log`; summary is `eval_summary.txt`.

Next required action: inspect whether the low LoRA eval reflects real reward/format overfitting or an evaluation/restore mismatch before using this as the comparison target for variants.
