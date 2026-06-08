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
| 2026-06-08 | | TPU VM | Verified updated W&B API key and reran saved smoke suite from `tests/smoke-tests/run_smoke_tests.py`. | Passed: W&B login verified, env/dataset/model/train/eval smoke stages completed. | Full baseline can use W&B, but keep smoke metrics separate from reportable results. |
| 2026-06-08 | | TPU VM | Ran `./scripts/run_tmux.sh` as-is. | Failed inside tmux: script tried to `cd` into `/home/boris_bolliet_cmbagent_community/tpu-2026/scripts`, which does not exist on this VM. | Update `run_tmux.sh` paths before using it for a full baseline run. |
| 2026-06-08 | | TPU VM | Ran `python evaluate.py` as-is from `scripts/` with venv and `.env` loaded. | Passed: default 64-example evaluation completed with `correct=33/64`, `acc=51.56%`, `partial=53.12%`, `format=6.25%`. | Treat this as a script validation result until base-vs-LoRA evaluation semantics are confirmed. |
| 2026-06-08 | | TPU VM | Patched `scripts/run_tmux.sh` after stale path failure. | Updated script to derive `REPO` from its own location, default `VENV` to `$HOME/venvs/tunix`, and source `~/.env` inside tmux before training. | Rerun `./scripts/run_tmux.sh` validation. |
| 2026-06-08 | | TPU VM | Reran patched `./scripts/run_tmux.sh`. | Path/venv/env loading worked: script used `/home/ext_felsomoye_gmail_com/tpu-2026`, logged into W&B/HF, and reached `train.py`. It then failed at `wandb.init` with `permission denied`. | Resolve W&B project/entity permissions before full baseline launch. |
| 2026-06-08 | | TPU VM | Added `WANDB_ENTITY=felsomoye-university-of-cambridge` and `WANDB_PROJECT=tunix` to `~/.env`, then reran W&B and `run_tmux.sh` validation. | Passed: W&B init/finish test succeeded, `run_tmux.sh` created a W&B run and started model/training setup. Session was intentionally stopped before a full baseline run. | Ready to launch the official baseline once run metadata is prepared. |
| 2026-06-08 | | TPU VM | Launched official baseline attempt `2026-06-08_baseline_seed42` from commit `2143844`. | Failed before training steps: TFDS failed constructing GSM8K from `./data/train` with `FieldDescriptor.label` AttributeError. W&B run `qvhm72qe` was created. | Resolve stale/incompatible TFDS cache or data path before relaunching baseline. |
| 2026-06-08 | | TPU VM | Cleared generated TFDS cache under `scripts/data/train` and `scripts/data/test`, then reran tiny dataset load using baseline paths. | Passed: TFDS rebuilt GSM8K cleanly, tiny train/test load returned lengths `train=1`, `val=0`, `test=1`, sample answer `13`. | Relaunch baseline from clean branch. |
| 2026-06-08 | | TPU VM | Relaunched official baseline after TFDS cache cleanup. | Failed again before training steps with same TFDS `FieldDescriptor.label` error from `./data/train`; W&B run `kcwsje77` was created. | Need a code/config-level data loading fix or dependency pin; cache deletion alone is insufficient. |
| 2026-06-08 | | TPU VM | Tested env-only protobuf workaround `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` on tiny TFDS load. | Failed: same `FieldDescriptor.label` error still occurs on existing `./data/train`; TFDS `TFRECORD` format is unsupported by `tfds.data_source`. | Need dependency pin or code/data-loading patch rather than env-only protobuf switch. |
| 2026-06-08 | | TPU VM | Downgraded protobuf in venv from `7.34.1` to `6.31.1`, then reran tiny TFDS load using baseline paths. | Passed: TFDS/GSM8K load succeeded with `tensorflow-datasets==4.9.9` and `protobuf==6.31.1`; requirements pin updated. | Validate `run_tmux.sh` startup, then relaunch baseline. |
| 2026-06-08 | | TPU VM | Ran `./scripts/run_tmux.sh` after protobuf pin as a startup validation. | Passed TFDS and entered GRPO training: W&B run `a6wiqxdb`, datasets `train=3364` and `val=374`. Session was intentionally stopped because the protobuf/docs changes were not committed yet. | Commit/push the protobuf pin, then relaunch a clean official baseline from the pushed branch. |

## Smoke Test Checklist

- [x] Model loads.
- [x] Dataset loads.
- [x] A few training steps run without error.
- [x] W&B or local logging works. W&B login now verifies successfully; smoke training still uses local TensorBoard-only metrics by design.
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

**Fix:** Resolved. Replaced `WANDB_API_KEY` with a valid W&B key; `wandb.login(..., verify=True)` succeeded for account `felsomoye` under entity `felsomoye-university-of-cambridge`.

**Impact:** Resolved environment issue. Full tracked runs can use W&B, but smoke-test metrics remain non-reportable.

### 2026-06-08 - Reused debug TFDS cache failed during training smoke

**Symptom:** A one-step training smoke test failed while constructing the TFDS `gsm8k` builder from `/tmp/content/debug_smoke_test_do_not_report/data/train` with `AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'label`.

**Cause:** The failure occurred when reusing the existing debug TFDS cache directory. Creating a fresh debug data directory made the same tiny TFDS load pass, so this appears to be isolated to the stale/incompatible debug cache rather than the GSM8K source generally.

**Fix:** Use fresh debug data/cache directories for smoke tests, or remove stale debug TFDS cache directories under `/tmp/content/debug_smoke_test_do_not_report` before rerunning. Full runs should use the normal configured data directory from a clean state.

**Impact:** Debug-only cache issue. It did not change model code, training logic, reward logic, or reported experiment results.

### 2026-06-08 - `run_tmux.sh` uses stale hard-coded paths

**Symptom:** Running `./scripts/run_tmux.sh` printed that it started tmux session `tunix`, but the session output showed `bash: line 1: cd: /home/boris_bolliet_cmbagent_community/tpu-2026/scripts: No such file or directory`.

**Cause:** `scripts/run_tmux.sh` contains hard-coded `REPO` and `VENV` paths for a different home directory/user.

**Fix:** Resolved in `scripts/run_tmux.sh`. The script now derives `REPO` from its own location, defaults `VENV` to `$HOME/venvs/tunix`, and sources `~/.env` inside the tmux-launched shell before calling `train.py`.

**Impact:** Original failed launch produced no model or experiment results. The path issue is fixed and validation reached `train.py`; launch is now blocked by W&B project/entity permissions.

### 2026-06-08 - W&B init permission denied during `run_tmux.sh` validation

**Symptom:** After patching `scripts/run_tmux.sh`, the script successfully started tmux, sourced the venv and `~/.env`, logged into W&B and Hugging Face, then failed in `train.py::maybe_init_wandb()` with `wandb.errors.errors.CommError: permission denied`.

**Cause:** W&B authentication works, but the configured `WANDB_ENTITY` / `WANDB_PROJECT` combination in `scripts/config.py` is not writable by the current W&B account. The account shown by W&B is `felsomoye` under entity `felsomoye-university-of-cambridge`, while the config default entity is `milindsarkaryt-iiser-mohali` unless overridden by environment.

**Fix:** Resolved for this TPU account. Added `WANDB_ENTITY=felsomoye-university-of-cambridge` and `WANDB_PROJECT=tunix` to `~/.env`; a W&B init/finish test succeeded and `run_tmux.sh` created a W&B run successfully.

**Impact:** Original validation run produced no model or experiment results. The W&B permission blocker is now resolved; the later `run_tmux.sh` validation was intentionally stopped before a full baseline run.

### 2026-06-08 - Official baseline attempt failed on TFDS cache

**Symptom:** Official baseline attempt `2026-06-08_baseline_seed42` launched with `./scripts/run_tmux.sh`, created W&B run `qvhm72qe`, downloaded the Gemma model, then failed before training steps while constructing GSM8K from `./data/train` with `AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'label`.

**Cause:** This matches the earlier stale/incompatible TFDS cache failure seen during smoke testing. The full run used the configured relative data path `./data/train` from inside `scripts/`, where `evaluate.py` had already created TFDS cache directories.

**Fix:** Not resolved. Removing `scripts/data/train` and `scripts/data/test` allowed a tiny dataset load to rebuild once, but the full `train.py` relaunch failed again with the same TFDS/protobuf error. Treat W&B runs `qvhm72qe` and `kcwsje77` as failed/invalid for experiment evidence.

**Impact:** No GRPO training steps completed and no checkpoint from either failed attempt should be reported. The issue remains an environment/data-loading blocker, not a model/reward/config result.

### 2026-06-08 - Protobuf env-only workaround did not fix TFDS

**Symptom:** Running the tiny TFDS load with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` still failed on `./data/train` with `AttributeError: 'FieldDescriptor' object has no attribute 'label`.

**Cause:** The installed stack is `tensorflow_datasets==4.9.9` and `protobuf==7.34.1`; the failure appears to be a package compatibility issue or TFDS metadata read issue, not only the compiled protobuf implementation.

**Fix:** Resolved for relaunch by downgrading/pinning `protobuf==6.31.1` in the venv and `requirements.txt`. Tiny TFDS/GSM8K load from the baseline paths now passes with `tensorflow-datasets==4.9.9`.

**Impact:** Dependency plumbing patch only. The default TFDS data source, GSM8K dataset, model, reward, and training configuration are unchanged.
