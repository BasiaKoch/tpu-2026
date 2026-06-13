# lora_rank128_alpha128_seed42

Changed vs. baseline: LoRA rank and alpha are both 128, and `NUM_GENERATIONS=8`.

## Outcome

- Status: completed.
- Start time: 2026-06-10T18:44:36Z.
- End time: 2026-06-11T03:16:55Z.
- Wall-clock: 8h 32m 19s.
- Final step: 3364.
- Exit code: 0.
- Final checkpoint: `/tmp/content/ckpts/actor/3364`.
- Preserved local checkpoint copy: `/home/funmilooi-somoye/tpu-2026/checkpoints/lora_rank128_alpha128_seed42/actor/`.
- W&B run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/aoz8dtkp.
- Final log marker: `Training finished.`

## Evaluation

- Base standalone greedy eval: `31/64`, `48.44%` accuracy, `51.56%` partial, `6.25%` format.
- Final W&B training-time eval at step `3364`: `rewards/eval/mean=1.575096`, `rewards/eval/score/mean=6.300384`, `rewards/eval/check_answer=0.773981`, `actor/eval/kl=0.512891`, `completions/eval/mean_length=460.006`.
- Standalone restored-checkpoint greedy accuracy: `19/64`, `29.69%` accuracy, `29.69%` partial, `56.25%` format.
- Evaluated from W&B artifact `felsomoye-university-of-cambridge/tunix/lora_rank128_alpha128_g2_seed42-actor:v0` at step `3364` using `scripts/evaluate.py` on CPU.

## Interpretation

- Final W&B eval reward is substantially healthier than the prior G=2 baseline final reward (`-0.240`), and there is no obvious final-step collapse in the summary metrics.
- This run changes both LoRA rank/alpha and group size (`RANK=128`, `ALPHA=128.0`, `NUM_GENERATIONS=8`), so it is not a clean single-variable group-size ablation.
- Standalone greedy accuracy from the restored checkpoint is below the base model on this split.
