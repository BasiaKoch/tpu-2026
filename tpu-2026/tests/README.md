# Smoke Tests

Small TPU bring-up checks for this repo. These are not benchmarks and their
accuracy/reward outputs must not be used as experiment results.

Run from the repo root on the TPU VM:

```bash
source ~/venvs/tunix/bin/activate
set -a
source ~/.env
set +a
python tests/smoke-tests/run_smoke_tests.py --stages all
```

The default run writes only under:

```text
/tmp/content/debug_smoke_test_do_not_report/
```

Stages:

- `env`: Python/JAX/dependency/credential visibility checks.
- `dataset`: tiny GSM8K train/test load.
- `model`: Gemma download/cache, base model load, LoRA wrapper, tokenizer load.
- `train`: one GRPO training step with TensorBoard-only metrics.
- `eval`: one-example evaluation with short generation.

You can run a subset:

```bash
python tests/smoke-tests/run_smoke_tests.py --stages env model
python tests/smoke-tests/run_smoke_tests.py --stages train --root /tmp/content/my_smoke
```

Notes:

- W&B is intentionally disabled for the training smoke. Full tracked runs still
  need a valid `WANDB_API_KEY`.
- The train/eval stages use fresh timestamped debug data directories to avoid
  stale TFDS cache issues.
- Keep smoke outputs separate from real run logs and checkpoints.
