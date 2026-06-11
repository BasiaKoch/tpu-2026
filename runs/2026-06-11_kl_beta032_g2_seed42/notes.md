# 2026-06-11_kl_beta032_g2_seed42

Changed vs. baseline: `BETA = 0.32` via environment override. Everything else was
held at the G2 baseline values in `kl-control-bk`: `NUM_GENERATIONS = 2`,
`TRAIN_MICRO_BATCH_SIZE = 1`, `LEARNING_RATE = 3e-6`, `RANK = 64`, `ALPHA = 64.0`,
`EPSILON = 0.2`, and `MAX_STEPS = 3364`.

This is the high-KL-control partner run to `kl_beta1e-6_g2_seed42`. W&B confirms the
run finished at step `3364`; the previous `docs/RUNS.md` entry incorrectly still
marked it as running.

Preserved actor checkpoints on TPU VM `basia`: `2000`, `2500`, `3000`, and `3364`.

W&B run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/oet2tfjd.

W&B checkpoint artifact: `felsomoye-university-of-cambridge/tunix/kl_beta032_g2_seed42_checkpoint:latest`.
Artifact upload run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/kfcc6z4m.

Evaluation: pending — run the standard greedy 64-example GSM8K eval on the preserved
step `3364` checkpoint, plus the retained sweep checkpoints if time allows.
