# Runs

Single source of truth for completed, failed, and planned runs.

| Run | Commit | Branch | Config | Seed | Steps | Wall-clock | Checkpoint | Eval split | Eval acc | Log link | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|---:|---|---|---|
| base_eval_seed42 | 7e696c4 | baseline-fls | default | 42 | n/a | | n/a | GSM8K test, 64-example greedy | 51.56% | `runs/2026-06-08_baseline_seed42/base_eval.log` | completed | Base model / zero-init LoRA wrapper: correct=33/64, partial=53.12%, format=6.25%. |
| baseline_seed42 | 2143844 | baseline-fls | default | 42 | 0 | | n/a | GSM8K test | | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/qvhm72qe; https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/kcwsje77 | failed | Failed before training during TFDS GSM8K dataset construction on initial launch and relaunch; see `runs/2026-06-08_baseline_seed42_failed/failure_summary*.txt`. |
| baseline_seed42 | 7e696c4 | baseline-fls | default | 42 | 3364 | 4h 41m 38s | `/tmp/content/ckpts/actor/3364` | GSM8K test, 64-example greedy | 3.12% | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl; `runs/2026-06-08_baseline_seed42/lora_eval.log` | completed | Patched baseline with `protobuf==6.31.1`; training finished successfully at step 3364 with exit code 0. Preserved copy under `checkpoints/baseline_seed42/actor/3364` and W&B artifact `felsomoye-university-of-cambridge/tunix/baseline_seed42_checkpoint:latest`. Eval: correct=2/64, partial=6.25%, format=12.50%. |
| reward_length_penalty_seed42 | | reward-variant | | 42 | | | | GSM8K test | | | planned | Reward shaping or length-control variant. |
| kl_beta_low_seed42 | | kl-control-variant | | 42 | | | | GSM8K test | | | planned | KL-control variant. |

## Status Values

Use `planned`, `running`, `completed`, `failed`, or `invalid`.

