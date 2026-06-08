# Baseline Patches

Record any changes needed to make the as-shipped baseline run. Keep this file focused on setup, path, dependency, and reproducibility fixes.

## Patch Log

| Date | Commit | File(s) | Problem | Patch | Impact on experiment |
|---|---|---|---|---|---|
| 2026-06-08 | n/a | `scripts/data` TFDS cache / configured data path | Official baseline failed before training with TFDS `FieldDescriptor.label` error from `./data/train`. | Removed generated TFDS cache directories `scripts/data/train` and `scripts/data/test`; verified clean rebuild with tiny dataset load. | Environment/data-cache only; no code/model/reward/config behaviour changed. |

## Rules

- Do not mix baseline patches with experimental changes.
- Explain whether the patch changes model behaviour, training data, reward calculation, or only environment/runtime plumbing.
- If the patch affects behaviour, the baseline must be treated as the patched baseline and documented clearly in `docs/RUNS.md`.

## Template

### `<date> - <patch-name>`

**Commit:** 

**Problem:** 

**Files changed:** 

**Patch summary:** 

**Behavioural impact:** 

**Verification:**

### 2026-06-08 - baseline-tfds-cache-failure

**Commit:** 2143844

**Problem:** Official baseline attempt failed before any training steps while constructing GSM8K from `./data/train` with a TFDS/protobuf `FieldDescriptor.label` error.

**Files changed:** None. Generated local TFDS cache directories under `scripts/data/train` and `scripts/data/test` were removed and rebuilt.

**Patch summary:** Removed stale generated TFDS cache under the configured relative data path. No tracked source files were changed for the cache cleanup.

**Behavioural impact:** Environment/data-cache only. The same GSM8K TFDS source and configured train/test paths are used.

**Verification:** Tiny dataset load from `scripts/` using `./data/train` and `./data/test` passed after cache cleanup: `train=1`, `val=0`, `test=1`, sample answer `13`.
