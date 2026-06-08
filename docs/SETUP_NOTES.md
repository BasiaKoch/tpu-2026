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
| 2026-06-08 | | TPU VM | Smoke test: Python/JAX/dependency/dataset/checkpoint/model-load checks. | Blocked at model load: Hugging Face returned 401 for gated Gemma repo because `HF_TOKEN` is not visible in the shell. | Create/populate `~/.env` or otherwise export `HF_TOKEN`, then rerun model-load and training smoke tests. |
| 2026-06-08 | | TPU VM | Tried to populate `~/.env` from Secret Manager secret `tunix-env`. | Failed: Secret Manager API is disabled for project `tpu-2026` (`SERVICE_DISABLED`). | Enable Secret Manager API or provide secrets manually in the TPU shell. |
| 2026-06-08 | | TPU VM | Tried to enable `secretmanager.googleapis.com` for `tpu-2026`. | Failed: active TPU compute service account lacks permission (`AUTH_PERMISSION_DENIED`). | Project owner/admin must enable the API or provide credentials another way. |
| 2026-06-08 | | TPU VM | Reran model-load smoke test after loading `~/.env`. | Blocked at model download: Hugging Face returned 403 because the token account is not authorized for `google/gemma-3-1b-it`. | Use a Hugging Face account/token that has accepted Gemma access, then rerun model-load smoke test. |
| 2026-06-08 | | TPU VM | Started one-step training smoke with local logging intended. | Failed before training: `wandb.errors.AuthenticationError`, W&B API key has 19 characters and must have 40+. | Replace `WANDB_API_KEY` with a valid key or disable W&B for smoke/full runs. |
| 2026-06-08 | | TPU VM | Reran model-load smoke test after Gemma access was granted. | Passed: Hugging Face snapshot download/cache, base model load, LoRA wrapper, and tokenizer load all succeeded on TPU. | Continue with training smoke. |
| 2026-06-08 | | TPU VM | Reran one-step training smoke with TensorBoard-only metrics and fresh debug data/cache dirs. | Passed: one GRPO training step completed, TensorBoard event file written, and Orbax actor checkpoint step `1` written. | Fix W&B key before full tracked runs. |
| 2026-06-08 | | TPU VM | Ran one-example evaluation smoke with reduced generation length. | Passed: evaluation script path completed on one GSM8K example. Accuracy numbers are smoke-only and not reportable. | Use full evaluation config only after baseline training. |

## Smoke Test Checklist

- [x] Model loads.
- [x] Dataset loads.
- [x] A few training steps run without error.
- [x] W&B or local logging works. Local TensorBoard logging works; W&B remains blocked by invalid `WANDB_API_KEY`.
- [x] Checkpoint path is valid and writable.
- [x] Evaluation script runs.

## Issues And Fixes

### `<date> - <short issue>`

**Symptom:** What failed?

**Cause:** What caused it, if known?

**Fix:** Exact command, config change, or code patch.

**Impact:** Does this affect reproducibility or reported results?

### 2026-06-08 - Hugging Face token not visible during smoke test

**Symptom:** Model-load smoke test failed in `scripts/model.py::download_weights()` with `huggingface_hub.errors.GatedRepoError: 401 Client Error` for `google/gemma-3-1b-it`.

**Cause:** The shell running the smoke test did not have `HF_TOKEN` set. A direct environment check also showed `WANDB_API_KEY`, `HF_TOKEN`, `KAGGLE_USERNAME`, and `KAGGLE_KEY` were all unset. No `~/.env` was visible to source from this account.

**Fix:** Pending. Populate `~/.env` from Secret Manager or manually export `HF_TOKEN` in the TPU shell, then rerun the model-load smoke test.

**Impact:** No training or evaluation results were produced. Environment-only blocker; no model/training behaviour has changed.

### 2026-06-08 - Secret Manager API disabled

**Symptom:** Attempting to run `gcloud secrets versions access latest --secret=tunix-env --project=tpu-2026 > ~/.env` failed with `PERMISSION_DENIED` / `SERVICE_DISABLED`.

**Cause:** `secretmanager.googleapis.com` is disabled for project `tpu-2026`. The active account reported by gcloud was the TPU compute service account.

**Fix:** Pending. Either enable Secret Manager API for `tpu-2026` and retry the documented `~/.env` restore command, or manually create/export the needed environment variables in the TPU shell.

**Impact:** Credentials are still unavailable, so model-load, W&B logging, and any gated Hugging Face download remain blocked. No code or experiment behaviour changed.

### 2026-06-08 - TPU service account cannot enable Secret Manager API

**Symptom:** Attempting to run `gcloud services enable secretmanager.googleapis.com --project=tpu-2026` failed with `PERMISSION_DENIED` / `AUTH_PERMISSION_DENIED`.

**Cause:** The active gcloud account is the TPU compute service account, and it does not have permission to enable Google Cloud services for the project.

**Fix:** Pending. A project owner/admin should enable Secret Manager API, or the team should provide `HF_TOKEN` and other required secrets directly in `~/.env` on the TPU VM.

**Impact:** The documented Secret Manager-based credential restore path cannot be completed from this TPU account. Smoke testing remains blocked at gated model download until `HF_TOKEN` is available.

### 2026-06-08 - Hugging Face token lacks Gemma repo access

**Symptom:** After loading `~/.env`, the model-load smoke test reached Hugging Face with credentials but failed with `huggingface_hub.errors.GatedRepoError: 403 Client Error` for `google/gemma-3-1b-it`.

**Cause:** The `HF_TOKEN` is present, but the Hugging Face account behind it is not on the authorized list for the gated Gemma model.

**Fix:** Resolved during smoke testing. Hugging Face Gemma access was granted for the token account, and the subsequent model-load smoke test passed.

**Impact:** Resolved environment/access issue. No code or experiment behaviour changed.

### 2026-06-08 - W&B API key invalid during training smoke

**Symptom:** A one-step training smoke test failed before model construction in `train.login_services()` with `wandb.errors.AuthenticationError: API key must have 40+ characters, has 19`.

**Cause:** `WANDB_API_KEY` is present in `~/.env`, but it is not a valid W&B API key length. `train.login_services()` calls `load_dotenv()` and then logs into W&B whenever `WANDB_API_KEY` is set.

**Fix:** Pending. Replace `WANDB_API_KEY` with a valid key from W&B, or leave it unset for smoke tests and rely on local TensorBoard/checkpoint logging.

**Impact:** External W&B logging is not validated yet. This does not affect model weights or training logic, but full tracked runs should not start until W&B is fixed or the team agrees to use local logging only.

### 2026-06-08 - Reused debug TFDS cache failed during training smoke

**Symptom:** A one-step training smoke test failed while constructing the TFDS `gsm8k` builder from `/tmp/content/debug_smoke_test_do_not_report/data/train` with `AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'label`.

**Cause:** The failure occurred when reusing the existing debug TFDS cache directory. Creating a fresh debug data directory made the same tiny TFDS load pass, so this appears to be isolated to the stale/incompatible debug cache rather than the GSM8K source generally.

**Fix:** Use fresh debug data/cache directories for smoke tests, or remove stale debug TFDS cache directories under `/tmp/content/debug_smoke_test_do_not_report` before rerunning. Full runs should use the normal configured data directory from a clean state.

**Impact:** Debug-only cache issue. It did not change model code, training logic, reward logic, or reported experiment results.
