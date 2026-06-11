# 2026-06-10_reward_length_g8bs1_seed42

Changed vs. the `n-generations-8` (G8) base: added the `length_penalty` reward function
to `scripts/rewards.py` (`LENGTH_TARGET = 600` chars, capped at `-1.0`) and appended it
to `REWARD_FNS`; and `TRAIN_MICRO_BATCH_SIZE` reduced 2 -> 1 because G8 + micro-batch-2
exhausts HBM on the single-chip v6e-1 (RESOURCE_EXHAUSTED after ~5h once completions get
long — see `docs/EXPERIMENTS.md`, `reward-length-on-g8`). Everything else at the G8-base
values: `NUM_GENERATIONS = 8`, `LEARNING_RATE = 3e-6`, `RANK = 64`, `BETA = 0.08`,
`EPSILON = 0.2`, data shuffle seed 42, `MAX_STEPS = 3364`.

Control: a G8 + bs=1 run with the identical config minus the length penalty, executed
on the same VM/venv immediately before this run; stopped at ~step 500+ to free the chip.
Its step-500 checkpoint is preserved at `/home/basiakoch/content/ckpts/actor/500` and
serves as the matched-step control (length-run step-500 checkpoint preserved at
`~/preserved/length/actor/500`). The control's uncommitted bs=1 edit was identical in
content to commit 89cde305, which post-dates that launch — noted for provenance honesty.

Comparison plan: matched-step accuracy comparison at step 500 (control vs. treatment,
only the length penalty differs); late-training behaviour (does the penalty prevent the
baseline-style length blowup / collapse?) read from the full trajectory curves vs.
baseline `jgs4c6kl` and the team's G8 runs. NOT directly comparable to Rowan's bs=2 G8
run because of the micro-batch change.

Hypothesis: the baseline collapse is length/format-driven (length blows up to ~471 then
craters to 0 with empty completions). A mild excess-length penalty keeps the policy away
from the verbose/degenerate attractor, holding within-group reward variance alive,
without adding KL drift.

Observed: pending — fill in from W&B run cyay16mj (completion length, eval reward, KL)
against the control and baseline.

Evaluation: pending — greedy 64-example suite (base + restored checkpoints: matched
step 500 + sweep 2000/2500/3000/3364) once the KL-control runs free the chip.

W&B checkpoint artifact: `felsomoye-university-of-cambridge/tunix/reward_length_g8bs1_seed42_checkpoint:latest`.
Artifact upload run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/f8g3j4ks.
