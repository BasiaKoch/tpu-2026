# Baseline Patches

Record any changes needed to make the as-shipped baseline run. Keep this file focused on setup, path, dependency, and reproducibility fixes.

## Patch Log

| Date | Commit | File(s) | Problem | Patch | Impact on experiment |
|---|---|---|---|---|---|
| 2026-06-08 | pending | `requirements.txt` | Official baseline failed before training with TFDS `FieldDescriptor.label` error from `./data/train`, including after cache cleanup. | Downgraded/pinned `protobuf==6.31.1`; tiny TFDS load now passes with default `tfds` source. | Dependency plumbing only; no model/data/reward/training config behaviour changed. |

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

**Files changed:** `requirements.txt`. Generated local TFDS cache directories were also cleaned during diagnosis.

**Patch summary:** Pinned `protobuf==6.31.1` instead of `protobuf==7.34.1` to restore compatibility with `tensorflow-datasets==4.9.9` metadata reads.

**Behavioural impact:** Environment/dependency plumbing only. The same default TFDS source, GSM8K split, model, rewards, and training configuration are used.

**Verification:** After installing `protobuf==6.31.1`, tiny dataset load from `scripts/` using `./data/train` and `./data/test` passed with `tensorflow-datasets==4.9.9`: `train=1`, `val=0`, `test=1`, sample answer `13`. A `run_tmux.sh` startup validation then passed TFDS and entered GRPO training with datasets `train=3364` and `val=374`; W&B run `a6wiqxdb` was stopped and treated as validation only because the pin was not committed yet.
