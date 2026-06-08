# Baseline Patches

Record any changes needed to make the as-shipped baseline run. Keep this file focused on setup, path, dependency, and reproducibility fixes.

## Patch Log

| Date | Commit | File(s) | Problem | Patch | Impact on experiment |
|---|---|---|---|---|---|
| 2026-06-08 | pending | `scripts/data` TFDS cache / configured data path | Official baseline failed before training with TFDS `FieldDescriptor.label` error from `./data/train`, including after cache cleanup. | Cache cleanup was insufficient; pending code/config-level data loading fix or dependency pin. | Must be documented as baseline plumbing; behavioural impact depends on final fix. |

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

**Patch summary:** Cache cleanup alone was insufficient. A code/config-level data loading fix or dependency pin is still pending.

**Behavioural impact:** Pending. The final fix should preserve the same GSM8K source and split to remain environment/plumbing-only.

**Verification:** Cache cleanup verification was insufficient: tiny dataset load passed, but full `train.py` relaunch failed again with the same TFDS error. Pending a stronger verification via successful baseline startup.
