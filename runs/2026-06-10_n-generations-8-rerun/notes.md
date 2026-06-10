# 2026-06-10_n-generations-8-rerun

Changed vs. baseline: `NUM_GENERATIONS = 8` and `LEARNING_RATE = 3e-6`. Everything else held constant.

Status: training completed successfully.

This run record was reconstructed from the committed run artifact plus the W&B export for run `4i8lcitv`. The exact GSM8K eval was re-run in this workspace with `JAX_PLATFORMS=cpu` because the TPU VM was not available here and the original TPU-side eval logs were not committed.

Run summary:
- Start: 2026-06-10T07:13:53+00:00
- End: 2026-06-10T14:32:20+00:00
- Wall-clock: 7h 18m 27s
- Final step: 3364
- Exit code: 0
- Final log marker: `Training finished.`
- Final checkpoint: `/tmp/content_n_generations_8_rerun/ckpts/actor/3364`
- Preserved local checkpoint copy: n/a

Evaluation completed with greedy decoding on the configured 64-example GSM8K test eval. CPU-fallback exact eval for the base model / zero-init LoRA wrapper: `31/64`, `48.44%` accuracy, `51.56%` partial, `6.25%` format. Restored LoRA checkpoint step `3364`: `38/64`, `59.38%` accuracy, `62.50%` partial, `95.31%` format.

W&B scalar summary at step 3364: `rewards/eval/mean=1.9081926804812832`, `rewards/eval/score/mean=7.632770721925134`, `completions/eval/mean_length=328.72493315508024`.

W&B run: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/4i8lcitv`
