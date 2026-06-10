# 2026-06-09_lr1e5_seed42

Changed vs. baseline: `LEARNING_RATE = 1e-5`. Everything else held constant.

Status: training completed successfully.

Run summary:
- Start: 2026-06-09T21:18:35Z
- End: 2026-06-10T05:21:57Z
- Wall-clock: 8h 03m 22s
- Final step: 3364
- Exit code: 0
- Final log marker: `Training finished.`
- Final checkpoint: `/tmp/content/ckpts/actor/3364`
- Preserved local checkpoint copy: `/home/funmilooi-somoye/tpu-2026/checkpoints/lr1e5_seed42/actor/3364`

Evaluation completed with greedy decoding on the configured 64-example GSM8K test eval. Base model / zero-init LoRA wrapper: `33/64`, `51.56%` accuracy, `53.12%` partial, `6.25%` format. Restored LoRA checkpoint step `3364`: `19/64`, `29.69%` accuracy, `34.38%` partial, `98.44%` format.

Checkpoint sweep: step `2000` reached `31.25%` accuracy, step `2500` reached `31.25%`, step `3000` reached `26.56%`, and final step `3364` reached `29.69%`.

W&B checkpoint artifact: `felsomoye-university-of-cambridge/tunix/lr1e5_seed42_checkpoint:latest`.

W&B artifact upload run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/ntnst5w9.

W&B: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/hozux9t6
