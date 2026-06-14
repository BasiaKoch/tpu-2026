Rerun of hard_medium_jsonl_g8_seed42 after W&B run hlo1w4go was interrupted by VM reboot.

Changed vs. failed attempt:
- Starts from scratch; no checkpoint was available to resume from hlo1w4go.
- Uses persistent TPU_CONTENT_DIR=/home/ext_felsomoye_gmail_com/tpu-2026/checkpoints/hard_medium_jsonl_g8_seed42_rerun_content instead of /tmp/content.
- Keeps DATA_SOURCE=jsonl and NUM_GENERATIONS=8.

W&B run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/6yiowy1y

Launch record, 2026-06-13:
- Started at 2026-06-13T15:26:45Z in tmux session tunix.
- W&B run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/6yiowy1y
- train.py PID 14647, last verified 2026-06-13T16:31:20Z.
- Persistent actor checkpoint directory: /home/ext_felsomoye_gmail_com/tpu-2026/checkpoints/hard_medium_jsonl_g8_seed42_rerun_content/ckpts/actor/
- actor/1 checkpoint exists as of 2026-06-13T16:31:20Z.


## Outcome

- Status: completed.
- Start time: 2026-06-13T15:26:45Z.
- End time: 2026-06-13T23:02:20Z.
- Wall-clock: 7h 35m 35s from local launch timestamp; W&B runtime 27,306.63s (~7h 35m 7s).
- Final step: 3364.
- Exit code: 0.
- Final checkpoint: `/home/ext_felsomoye_gmail_com/tpu-2026/checkpoints/hard_medium_jsonl_g8_seed42_rerun_content/ckpts/actor/3364`.
- Retained checkpoints: `1`, `500`, `1000`, `1500`, `2000`, `2500`, `3000`, `3364`.
- W&B checkpoint artifact: `felsomoye-university-of-cambridge/tunix/hard_medium_jsonl_g8_seed42_rerun_checkpoint:latest`.
- W&B artifact upload run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/336gnz8g.
- Final log marker: `Training finished.`

Final W&B summary at completion included `rewards/eval/mean=1.481680`, `rewards/eval/score/mean=5.926721`, `actor/eval/kl=0.279609`, and `completions/eval/mean_length=324.882353`.

Next required action: run the standard GSM8K 64-example greedy base and restored-checkpoint eval, including retained checkpoint sweep steps `2000`, `2500`, `3000`, and `3364`.
