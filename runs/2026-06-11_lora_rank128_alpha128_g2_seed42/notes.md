# lora_rank128_alpha128_g2_seed42

Changed vs. baseline: LoRA rank and alpha are both 128, and `NUM_GENERATIONS=2`.

## Outcome

- Status: completed.
- Start time: 2026-06-11T11:14:00Z.
- End time: 2026-06-11T16:43:09Z.
- Wall-clock: 5h 29m 9s.
- Final step: 3364.
- Exit code: 0.
- Final checkpoint: `/tmp/content/ckpts/actor/3364`.
- Preserved local checkpoint copy: `/home/funmilooi-somoye/tpu-2026/checkpoints/lora_rank128_alpha128_g2_seed42/actor/`.
- W&B run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/v5cvlwkm.
- W&B artifact: `felsomoye-university-of-cambridge/tunix/lora_rank128_alpha128_g2_seed42-actor:v0`.
- Final log marker: `Training finished.`

## Evaluation

- Base standalone greedy eval: `34/64`, `53.12%` accuracy, `54.69%` partial, `7.81%` format.
- Final W&B eval at step `3364`: `rewards/eval/mean=0.8610461229946524`, `rewards/eval/score/mean=3.4441844919786098`, `rewards/eval/check_answer=0.3125`, `actor/eval/kl=0.668775737285614`, `completions/eval/mean_length=274.9438502673797`.
- Standalone restored-checkpoint greedy eval: `21/64`, `32.81%` accuracy, `32.81%` partial, `53.12%` format.
