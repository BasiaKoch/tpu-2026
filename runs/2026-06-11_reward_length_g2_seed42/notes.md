# 2026-06-11_reward_length_g2_seed42

Changed vs. the G2 baseline: added the `length_penalty` reward function to
`scripts/rewards.py` (`LENGTH_TARGET = 600` chars, capped at `-1.0`) and appended it
to `REWARD_FNS`. Everything else remained at the G2 baseline values in this branch:
`NUM_GENERATIONS = 2`, `TRAIN_MICRO_BATCH_SIZE = 1`, `LEARNING_RATE = 3e-6`,
`RANK = 64`, `ALPHA = 64.0`, `BETA = 0.08`, `EPSILON = 0.2`, and `MAX_STEPS = 3364`.

The initial preservation commands were accidentally run on the local Mac shell, where
`/tmp/content` and `~/tpu-2026/scripts/train.log` did not exist. The actual artifacts
were then found on TPU VM `basia` (`t1v-n-cf158ba8-w-0`, project
`basia-tpu-06092034`, zone `us-east5-b`) and copied from volatile `/tmp/content` into
`/home/basiakoch/preserved/length_g2_content`.

Preserved actor checkpoints: `2000`, `2500`, `3000`, and `3364`.

W&B run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jcp0b5cy.

W&B checkpoint artifact: `felsomoye-university-of-cambridge/tunix/reward_length_g2_seed42_checkpoint:latest`.
Artifact upload run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/gy146naa.

Evaluation: pending — run the standard greedy 64-example GSM8K eval on the preserved
step `3364` checkpoint, plus the retained sweep checkpoints if time allows.
