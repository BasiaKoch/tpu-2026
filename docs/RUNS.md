# Runs

Single source of truth for completed, failed, and planned runs.

| Run | Commit | Branch | Config | Seed | Steps | Wall-clock | Checkpoint | Eval split | Eval acc | Log link | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|---:|---|---|---|
| base_eval_seed42 | | baseline-reproduction | default | 42 | n/a | | n/a | GSM8K test | | | planned | Base model only. |
| baseline (`jgs4c6kl`, leafy-microwave-6) | | tunix | default, `NUM_GENERATIONS=2` | not logged | 3364 | ~4.7 h (16,896 s) | step 448 (best); final not recommended | GSM8K test | eval reward mean -0.240 (final); **1.711 @ step 448 (best)** | [wandb](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl) | completed | G=2 baseline. Learned early then collapsed (KL spike 41, grad-norm 696, completions → 0 length). Use step ~448, not final. See [baseline-analysis.md](rowan/baseline-analysis.md). |
| group_size_g8 | | TBD | default, `NUM_GENERATIONS=8` | | 3364 | | | GSM8K test | | | running | G=8 group-size experiment. See [EXPERIMENTS.md](EXPERIMENTS.md) `group-size-g8`. |
| g8_bs1_control_seed42 | 89cde30 (content-equivalent; edit uncommitted at launch) | n-generations-8 + bs=1 edit | `NUM_GENERATIONS=8`, `TRAIN_MICRO_BATCH_SIZE=1` | 42 | ~500+ (stopped) | | `/home/basiakoch/content/ckpts/actor/500` (Barbara's VM) | GSM8K test, 64-example greedy | eval pending | (W&B auto-named; same project) | invalid | Deliberately stopped to free the chip; step-500 checkpoint retained as the matched-step control for the length variant. Not a full run. |
| reward_length_g8bs1_seed42 | 89cde30 | reward-length-on-g8-bk | G8 + bs=1 + `length_penalty` reward | 42 | 3364 | ~7h (approx; see W&B) | `~/preserved/length_final/actor/{2000,2500,3000,3364}`; matched-step `~/preserved/length/actor/500` (Barbara's VM) | GSM8K test, 64-example greedy | eval pending | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/cyay16mj | completed | Length penalty on the G8+bs1 base; control = g8_bs1_control step-500 checkpoint. Supersedes the planned `reward_length_penalty_seed42`. See `runs/2026-06-10_reward_length_g8bs1_seed42/`. |
| kl_beta_low_seed42 | | kl-control-variant | | 42 | | | | GSM8K test | | | planned | KL-control variant. |

## Status Values

Use `planned`, `running`, `completed`, `failed`, or `invalid`.

