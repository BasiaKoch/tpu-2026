# Baseline k=2 Seed 42 to 5864

Launched: 2026-06-17 16:24:58 UTC

Completed: 2026-06-17 23:56:42 UTC (exit code 0, step 5864/5864)

This run reruns the baseline hyperparameters on branch `baseline-k=2`, extending
training from the prior derived `MAX_STEPS=3364` to `MAX_STEPS=5864`.

## Launch

- tmux session: `tunix`
- process: `python -u train.py`
- working directory: `/home/funmilooi-somoye/tpu-2026/scripts`
- command:

```bash
cd /home/funmilooi-somoye/tpu-2026/scripts
source /home/funmilooi-somoye/venvs/tunix/bin/activate
if [ -f ~/.env ]; then set -a; source ~/.env; set +a; fi
python -u train.py
```

## Source State

- branch: `baseline-k=2`
- commit at launch: `3a22268d1762f628cc7f17761dc0a30dc55b05db`
- active config snapshot: `config.py` in this folder
- baseline reward file: unchanged relative to `scripts/baseline_scripts/rewards.py`

## Key Config

- `SEED=42`
- `DATA_SOURCE=tfds`
- `NUM_GENERATIONS=2`
- `BETA=0.08`
- `EPSILON=0.2`
- `TRAIN_MICRO_BATCH_SIZE=1`
- `NUM_BATCHES=6516`
- `TRAIN_FRACTION=0.9`
- `NUM_EPOCHS=1`
- `MAX_STEPS=5864`
- `LEARNING_RATE=3e-6`
- `MAX_GRAD_NORM=0.1`
- `TEMPERATURE=0.9`
- `TOP_P=1.0`
- `TOP_K=50`
- `TOTAL_GENERATION_STEPS=768`

## Observed Startup

- W&B run id: `keh7f5es`
- W&B run name: `genial-capybara-52`
- W&B URL: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/keh7f5es`
- local W&B dir: `/home/funmilooi-somoye/tpu-2026/scripts/wandb/run-20260617_162508-keh7f5es`
- trainer printed: `Datasets: train=5864 val=652`
- trainer printed: `Starting GRPO training. CKPT_DIR=/tmp/content/ckpts/  MAX_STEPS=5864`

## Notes

The only intentional hyperparameter change from the active baseline defaults is
`NUM_BATCHES: 3738 -> 6516`, which makes the existing derived formula evaluate
to `MAX_STEPS=5864`.

The final checkpoint is `/tmp/content/ckpts/actor/5864` and was uploaded to
W&B as `baseline_k2_seed42_to5864_checkpoint:v0`. TensorBoard logs were
uploaded as `baseline_k2_seed42_to5864_tensorboard:v0`. Standalone baseline
evaluation remains pending.
