# Baseline Patches

Record any changes needed to make the as-shipped baseline run. Keep this file focused on setup, path, dependency, and reproducibility fixes.

## Patch Log

| Date | Commit | File(s) | Problem | Patch | Impact on experiment |
|---|---|---|---|---|---|
| | | | | | |

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

