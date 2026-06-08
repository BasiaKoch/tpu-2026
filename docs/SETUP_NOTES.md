# Setup Notes

Record environment setup, deviations from the course instructions, and fixes discovered during smoke tests or full runs.

## Environment

| Item | Value |
|---|---|
| TPU name | `boris` |
| Project ID | `tpu-2026` |
| Zone | `us-east5-a` |
| Accelerator type | `v6e-1` |
| Python | `3.12` |
| Main setup guide | `tpu-setup.md` |

## Setup Log

| Date | Person | Machine/TPU | Action | Result | Follow-up |
|---|---|---|---|---|---|
| 2026-06-08 | | | Created repo tracking folders and docs. | Done. | Fill in TPU-specific setup once run. |

## Smoke Test Checklist

- [ ] Model loads.
- [ ] Dataset loads.
- [ ] A few training steps run without error.
- [ ] W&B or local logging works.
- [ ] Checkpoint path is valid and writable.
- [ ] Evaluation script runs.

## Issues And Fixes

### `<date> - <short issue>`

**Symptom:** What failed?

**Cause:** What caused it, if known?

**Fix:** Exact command, config change, or code patch.

**Impact:** Does this affect reproducibility or reported results?

