# Runs

Single source of truth for completed, failed, and planned runs.

| Run | Commit | Branch | Config | Seed | Steps | Wall-clock | Checkpoint | Eval split | Eval acc | Log link | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|---:|---|---|---|
| base_eval_seed42 | | baseline-fls | default | 42 | n/a | | n/a | GSM8K test | | | planned | Base model only. |
| baseline_seed42 | | baseline-fls | default | 42 | | | | GSM8K test | | | planned | Default GRPO baseline. |
| reward_length_penalty_seed42 | | reward-variant | | 42 | | | | GSM8K test | | | planned | Reward shaping or length-control variant. |
| kl_beta_low_seed42 | | kl-control-variant | | 42 | | | | GSM8K test | | | planned | KL-control variant. |

## Status Values

Use `planned`, `running`, `completed`, `failed`, or `invalid`.

