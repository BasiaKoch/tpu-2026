# Runs

Single source of truth for completed, failed, and planned runs.

| Run | Commit | Branch | Config | Seed | Steps | Wall-clock | Checkpoint | Eval split | Eval acc | Log link | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|---:|---|---|---|
| base_eval_seed42 | 7e696c4 | baseline-fls | default | 42 | n/a | | n/a | GSM8K test, 64-example greedy | 51.56% | `runs/2026-06-08_baseline_seed42/base_eval.log` | completed | Base model / zero-init LoRA wrapper: correct=33/64, partial=53.12%, format=6.25%. |
| baseline_seed42 | 2143844 | baseline-fls | default | 42 | 0 | | n/a | GSM8K test | | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/qvhm72qe; https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/kcwsje77 | failed | Failed before training during TFDS GSM8K dataset construction on initial launch and relaunch; see `runs/2026-06-08_baseline_seed42_failed/failure_summary*.txt`. |
| baseline_seed42 (`jgs4c6kl`, leafy-microwave-6) | 7e696c4 | baseline-fls / tunix | default, `NUM_GENERATIONS=2` | 42 | 3364 | 4h 41m 38s (~16,896 s) | `/tmp/content/ckpts/actor/3364`; best eval around step 448; final not recommended | GSM8K test, 64-example greedy | 3.12%; eval reward mean -0.240 (final), **1.711 @ step 448 (best)** | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl; `runs/2026-06-08_baseline_seed42/lora_eval.log` | completed | Patched baseline with `protobuf==6.31.1`; training finished successfully at step 3364 with exit code 0. Preserved copy under `checkpoints/baseline_seed42/actor/3364` and W&B artifact `felsomoye-university-of-cambridge/tunix/baseline_seed42_checkpoint:latest`. Eval: correct=2/64, partial=6.25%, format=12.50%. G=2 baseline learned early then collapsed (KL spike 41, grad-norm 696, completions -> 0 length). Use step ~448, not final. See [baseline-analysis.md](rowan/baseline-analysis.md). |
| lr1e5_seed42 (`hozux9t6`, wobbly-gorge-22) | 99059ce | learning-rate-fls | `LEARNING_RATE=1e-5`, otherwise baseline | 42 | 3364 | 8h 03m 22s (~29,002 s) | `/tmp/content/ckpts/actor/3364`; preserved `checkpoints/lr1e5_seed42/actor/3364`; W&B artifact `felsomoye-university-of-cambridge/tunix/lr1e5_seed42_checkpoint:latest` | GSM8K test, 64-example greedy | final 29.69%; best retained 31.25% @ steps 2000/2500 | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/hozux9t6; artifact https://wandb.ai/felsomoye-university-of-cambridge/tunix/artifacts/model/lr1e5_seed42_checkpoint/latest; `runs/2026-06-09_lr1e5_seed42/eval_summary.txt` | completed | Training finished successfully at step 3364 with exit code 0. Base eval: correct=33/64, acc=51.56%. Final LoRA eval: correct=19/64, acc=29.69%, partial=34.38%, format=98.44%. Higher LR strongly improved format adherence but still underperformed base numeric accuracy. |
| group_size_g8 | | TBD | default, `NUM_GENERATIONS=8` | | 3364 | | | GSM8K test | | | running | G=8 group-size experiment. See [EXPERIMENTS.md](EXPERIMENTS.md) `group-size-g8`. |
| reward_length_penalty_seed42 | | reward-variant | | 42 | | | | GSM8K test | | | planned | Reward shaping or length-control variant. |
| kl_beta_low_seed42 | | kl-control-variant | | 42 | | | | GSM8K test | | | planned | KL-control variant. |

## Status Values

Use `planned`, `running`, `completed`, `failed`, or `invalid`.
