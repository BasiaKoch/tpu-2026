# baseline_seed42

Current patched baseline run.

The earlier baseline attempts from this date were moved to `runs/2026-06-08_baseline_seed42_failed/`. Those attempts failed before training during TFDS GSM8K dataset construction.

This run uses commit `7e696c428687bc083604690f5f51009da6abb6d9`, which pins `protobuf==6.31.1` to fix the TFDS/protobuf failure. No model, dataset, reward, or training hyperparameter changes are intended beyond that dependency/runtime patch.

Live stdout/stderr is written to `scripts/train.log`. This folder stores pointers and reproducibility metadata while the run is active.
