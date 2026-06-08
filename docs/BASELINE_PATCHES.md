# Baseline Patches

Record any changes needed to make the as-shipped baseline run. Keep this file focused on setup, path, dependency, and reproducibility fixes.

## Patch Log

| Date | Commit | File(s) | Problem | Patch | Impact on experiment |
|---|---|---|---|---|---|
| 2026-06-08 | pending | `scripts/data` TFDS cache / configured data path | Official baseline failed before training with TFDS `FieldDescriptor.label` error from `./data/train`. | Pending: clean stale cache or move to a fresh data path before relaunch. | Environment/data-cache only if fixed without changing dataset contents. |

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

**Files changed:** Pending.

**Patch summary:** Pending. Likely options are to remove the stale TFDS cache under the configured data path or point the run at a fresh data directory.

**Behavioural impact:** Should be environment/data-cache only if the same GSM8K TFDS source and split are used.

**Verification:** Pending relaunch after cache/path cleanup.
