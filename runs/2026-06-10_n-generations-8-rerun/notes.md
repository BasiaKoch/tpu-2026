# 2026-06-10 n-generations-8-rerun

Completed G=8 / LR=3e-6 rerun from branch `n-generations-8-rerun`.

Config snapshot captured before launch.

Key settings:
- `NUM_GENERATIONS = 8`
- `LEARNING_RATE = 3e-6`
- `SEED = 42`

## Outcome

- Status: completed and evaluated.
- Start time: 2026-06-10T07:13:53Z.
- End time: 2026-06-10T14:32:17Z.
- Wall-clock: 7h 18m 24s.
- Final step: 3364.
- Exit code: 0.
- Final checkpoint: `/tmp/content_n_generations_8_rerun/ckpts/actor/3364`.
- Sweep checkpoints: `/tmp/content_n_generations_8_rerun/ckpts/actor/{2000,2500,3000,3364}`.
- W&B run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/4i8lcitv.

Evaluation used greedy decoding on the configured 64-example GSM8K test eval.
Base model / zero-init LoRA wrapper: `31/64`, `48.44%` accuracy, `51.56%` partial, `6.25%` format.
Restored LoRA checkpoint step `3364`: `38/64`, `59.38%` accuracy, `62.50%` partial, `95.31%` format.

Next required action: keep this branch as the record for the `G=8, LR=3e-6` run and move the next experiment to the rank study (`RANK=128`, `ALPHA=128`).
