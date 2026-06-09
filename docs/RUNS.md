# Runs

Single source of truth for completed, failed, and planned runs.

| Run | Commit | Branch | Config | Seed | Steps | Wall-clock | Checkpoint | Eval split | Eval acc | Log link | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|---:|---|---|---|
| base_eval_seed42 | | baseline-reproduction | default | 42 | n/a | | n/a | GSM8K test | | | planned | Base model only. |
| baseline (`jgs4c6kl`, leafy-microwave-6) | | tunix | default, `NUM_GENERATIONS=2` | not logged | 3364 | ~4.7 h (16,896 s) | step 448 (best); final not recommended | GSM8K test | eval reward mean -0.240 (final); **1.711 @ step 448 (best)** | [wandb](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl) | completed | G=2 baseline. Learned early then collapsed (KL spike 41, grad-norm 696, completions → 0 length). Use step ~448, not final. See [baseline-analysis.md](rowan/baseline-analysis.md). |
| group_size_g8 | | TBD | default, `NUM_GENERATIONS=8` | | 3364 | | | GSM8K test | | | running | G=8 group-size experiment. See [EXPERIMENTS.md](EXPERIMENTS.md) `group-size-g8`. |
| reward_length_penalty_seed42 | | reward-variant | | 42 | | | | GSM8K test | | | planned | Reward shaping or length-control variant. |
| kl_beta_low_seed42 | | kl-control-variant | | 42 | | | | GSM8K test | | | planned | KL-control variant. |

## Status Values

Use `planned`, `running`, `completed`, `failed`, or `invalid`.

