# Evidence: Funmi (FunmiLS) Experiments, 2026-06-12

> **Auditor note:** I am an AI auditor with no personal memory of branches, runs, W&B projects, or TPU VMs. All statements below are derived exclusively from local repo evidence: git history, run metadata files, eval logs, config snapshots, and docs. W&B was **not queried** — URLs are recorded from local metadata only.

---

## Summary

One completed GRPO baseline training run (`baseline_seed42`) exists on branch `baseline-fls`, plus two earlier failed attempts that never began training. The successful run completed 3364 GRPO steps in ~4h 42m and was evaluated at four retained checkpoints (steps 2000, 2500, 3000, 3364) plus the base model. **The trained model degrades below the base model** — the best retained checkpoint (step 2000) scores 28.12% vs. 51.56% for the base. The final checkpoint scores 3.12%. This appears to be a real training/checkpoint-selection failure, not a restore bug.

No variant experiments have been completed on this branch. Two variants (reward-length-penalty, KL-control) are listed as `planned` in `docs/RUNS.md`.

No hard-question mining was used.

---

## Git State

| Item | Value |
|---|---|
| Repo path | `/Users/funmilooi-somoye/Desktop/baseline agent/tpu-2026` |
| Current branch | `baseline-fls` |
| Current HEAD | `af08452d97a296b5f1e9038bda04251891c0b78e` — "Add I.1 baseline results and artifacts" |
| Tracking | `origin/baseline-fls` (up to date) |
| Dirty worktree | Clean — `git status --short --branch` shows `## baseline-fls...origin/baseline-fls` only |
| Relevant remote branches | `origin/baseline-fls`, `origin/improvement-rd`, `origin/main`, plus 15 other team branches |

### Relevant commits on `baseline-fls`

| Hash | Date (UTC) | Message |
|---|---|---|
| `af08452` | 2026-06-09 09:33 | Add I.1 baseline results and artifacts |
| `a884b61` | 2026-06-08 21:26 | Log current baseline run metadata |
| `7e696c4` | 2026-06-08 18:14 | Pin protobuf for TFDS baseline |
| `8fadc5b` | 2026-06-08 18:04 | Record failed protobuf TFDS workaround |
| `4607641` | 2026-06-08 17:59 | Record failed baseline relaunch |
| `8eaaec4` | 2026-06-08 17:54 | Document TFDS cache cleanup |
| `97881b8` | 2026-06-08 17:47 | Record failed baseline launch |
| `2143844` | 2026-06-08 17:35 | Prepare baseline run metadata |
| `3ec8727` | 2026-06-08 17:31 | Record successful tmux startup validation |
| `5eec0dd` | 2026-06-08 17:09 | Fix tmux launcher paths |

---

## Experiment Register

| Run | Branch | Commit | Estimator | K | Seed | Steps | Status | Best exact | Final exact | W&B | Artefacts usable? |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|---|
| baseline_seed42 (failed x2) | baseline-fls | `2143844` | GRPO | 2 | 42 | 0 | failed | — | — | [qvhm72qe](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/qvhm72qe), [kcwsje77](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/kcwsje77) | No |
| baseline_seed42 (successful) | baseline-fls | `7e696c4` | GRPO | 2 | 42 | 3364 | completed | 28.12% (step 2000) | 3.12% (step 3364) | [jgs4c6kl](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl) | Yes — with caveats |

---

## Run Details

### Failed baseline attempts (pre-training failures)

#### 1. Identity

| Item | Value |
|---|---|
| Run nickname | `baseline_seed42_failed` |
| Run ID | W&B: `qvhm72qe` (attempt 1), `kcwsje77` (attempt 2) |
| Owner/name | FunmiLS / `f.looisomoye@outlook.com` |
| Team/fork | `BasiaKoch/tpu-2026` (inferred from `git remote`) |
| Branch | `baseline-fls` |
| Commit | `2143844b5bb279299e9071e8e4ab11616e7a4f33` |
| Dirty worktree at launch | Clean — `git_status_before.txt` says "nothing to commit, working tree clean" |
| W&B project/entity | `tunix` / `felsomoye-university-of-cambridge` |
| Local run root | `runs/2026-06-08_baseline_seed42_failed/` |
| Date/time launched | 2026-06-08 17:35:09 UTC |

#### 2. Configuration

Same as the successful run below (identical `config_snapshot.py`). NUM_GENERATIONS=2, seed 42, GRPO.

#### 3. Runtime and Platform

| Item | Value |
|---|---|
| TPU type | v6e-1 (from `docs/I1_BASELINE_RESULTS.md`) |
| Attempt 1 | Started 17:35:09 UTC, failed at 17:45:55 UTC (~10 min) |
| Attempt 2 (relaunch) | Failed at 17:59:00 UTC (~13 min after original start) |
| Completed | **No** — both failed before training |
| Failure reason | `AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'label'` — TFDS/protobuf incompatibility when constructing GSM8K dataset |

#### 4. Artefacts

No checkpoints, no TensorBoard events, no eval CSVs. Only failure summaries and metadata files.

#### 5. Metrics

None — no training steps executed.

#### 6. Caveats

- **Not usable** for any reporting purpose. These runs produced zero training steps.

---

### Baseline K=2 Run (successful)

#### 1. Identity

| Item | Value |
|---|---|
| Run nickname | `baseline_seed42` |
| Run ID (W&B) | `jgs4c6kl` |
| Owner/name | FunmiLS / `f.looisomoye@outlook.com` |
| Team/fork | `BasiaKoch/tpu-2026` |
| Branch | `baseline-fls` |
| Commit | `7e696c428687bc083604690f5f51009da6abb6d9` ("Pin protobuf for TFDS baseline") |
| Dirty worktree at launch | **Yes (minor)** — `git_status_before.txt` shows `M docs/BASELINE_PATCHES.md`, `M docs/RUNS.md`, `M docs/SETUP_NOTES.md`. These are documentation files only; no training code was dirty. |
| W&B URL | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl |
| W&B project/entity | `tunix` / `felsomoye-university-of-cambridge` |
| W&B checkpoint artifact | `felsomoye-university-of-cambridge/tunix/baseline_seed42_checkpoint:latest` |
| W&B artifact upload run | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/bah3rczo |
| Local run root | `runs/2026-06-08_baseline_seed42/` |
| Preserved checkpoint (on TPU VM) | `/home/ext_felsomoye_gmail_com/tpu-2026/checkpoints/baseline_seed42/actor/3364` |
| Date/time launched | 2026-06-08T18:17:09Z |

#### 2. Configuration

| Parameter | Value |
|---|---|
| Estimator | GRPO (`tunix.rl.grpo.grpo_learner.GRPOLearner`) |
| NUM_GENERATIONS (K) | 2 |
| Seed | 42 (data shuffle in `data.py`) |
| MAX_STEPS | `int(3738 * 1 * 0.9 * 1) = 3364` |
| SAVE_INTERVAL_STEPS | 500 |
| MAX_TO_KEEP | 4 |
| EVAL_EVERY_N_STEPS | 64 |
| NUM_ITERATIONS (mu) | 1 |
| BETA (KL penalty) | 0.08 |
| EPSILON (PPO clip) | 0.2 |
| LEARNING_RATE | 3e-6 |
| WARMUP_STEPS | 0.1 * 3364 = 336.4 |
| MAX_GRAD_NORM | 0.1 |
| TRAIN_MICRO_BATCH_SIZE | 1 |
| NUM_BATCHES | 3738 |
| NUM_TEST_BATCHES | 64 |
| TRAIN_FRACTION | 0.9 |
| NUM_EPOCHS | 1 |
| LoRA RANK | 64 |
| LoRA ALPHA | 64.0 |
| Model | `google/gemma-3-1b-it` |
| TEMPERATURE (rollout) | 0.9 |
| TOP_K (rollout) | 50 |
| TOP_P (rollout) | 1.0 |
| MAX_PROMPT_LENGTH | 256 |
| TOTAL_GENERATION_STEPS | 768 |
| Dataset/source | TFDS GSM8K (`tfds`) |
| CKPT_DIR | `/tmp/content/ckpts/` |
| TENSORBOARD_DIR | `/tmp/content/tmp/tensorboard/grpo` |
| Hard-question manifest | Not used |
| Reward functions | `match_format_exactly`, `match_format_approximately`, `check_answer`, `check_numbers` (standard — no changes from repo default) |
| Decoding/eval preset | `greedy` (temperature=None, top_k=1, top_p=None) |
| W&B entity (config default) | `milindsarkaryt-iiser-mohali` (but overridden at runtime to `felsomoye-university-of-cambridge` per W&B URL evidence) |

#### 3. Runtime and Platform

| Item | Value |
|---|---|
| TPU type | v6e-1 |
| TPU zone | Not recorded locally |
| TPU VM account | `ext_felsomoye_gmail_com` (from checkpoint paths) |
| Start time | 2026-06-08T18:17:09Z |
| End time | 2026-06-08T22:58:47Z |
| Wall-clock duration | 4h 41m 38s |
| Run completed | **Yes** — exit code 0, final log marker "Training finished." |
| Final step | 3364 |
| tmux session | `tunix` (PID 49236) |
| Training PID | 49244 |

#### 4. Artefacts

| Artefact | Path | Present? |
|---|---|---|
| Run metadata dir | `runs/2026-06-08_baseline_seed42/` | ✅ (22 files) |
| Config snapshot | `runs/2026-06-08_baseline_seed42/config_snapshot.py` | ✅ (matches `scripts/config.py`) |
| Training log (live) | `scripts/train.log` (referenced as `../../scripts/train.log`) | ⚠️ Not found locally — was on TPU VM at `/home/ext_felsomoye_gmail_com/tpu-2026/scripts/train.log`, ~19 MB at ledger creation |
| TensorBoard events | `/tmp/content/tmp/tensorboard/grpo/events.out.tfevents.1780942641.t1v-n-f769470f-w-0` | ⚠️ Was on TPU VM `/tmp/` (volatile); **not present locally** |
| Checkpoint (final, on TPU) | `/tmp/content/ckpts/actor/3364` | ⚠️ Not available locally — was in `/tmp/` on TPU |
| Checkpoint (preserved, on TPU) | `/home/ext_felsomoye_gmail_com/tpu-2026/checkpoints/baseline_seed42/actor/3364` | ⚠️ Not found locally — on TPU VM persistent storage |
| Checkpoint (W&B artifact) | `felsomoye-university-of-cambridge/tunix/baseline_seed42_checkpoint:latest` | ✅ Uploaded to W&B (not verified, W&B not queried) |
| Base eval log | `runs/2026-06-08_baseline_seed42/base_eval.log` | ✅ |
| LoRA eval log (step 3364) | `runs/2026-06-08_baseline_seed42/lora_eval.log` | ✅ |
| LoRA eval log (step 2000) | `runs/2026-06-08_baseline_seed42/lora_eval_step2000.log` | ✅ |
| LoRA eval log (step 2500) | `runs/2026-06-08_baseline_seed42/lora_eval_step2500.log` | ✅ |
| LoRA eval log (step 3000) | `runs/2026-06-08_baseline_seed42/lora_eval_step3000.log` | ✅ |
| Eval summary | `runs/2026-06-08_baseline_seed42/eval_summary.txt` | ✅ |
| Degradation diagnostic | `runs/2026-06-08_baseline_seed42/degradation_diagnostic.txt` | ✅ |
| Completion summary | `runs/2026-06-08_baseline_seed42/completion_summary.txt` | ✅ |
| Notes | `runs/2026-06-08_baseline_seed42/notes.md` | ✅ |
| Training curves (CSV) | `report_assets/baseline_mean_reward_curve.csv`, `report_assets/baseline_kl_curve.csv` | ✅ |
| Training curves (PNG/PDF) | `report_assets/baseline_mean_reward_curve.{png,pdf}`, `report_assets/baseline_kl_curve.{png,pdf}` | ✅ |
| Curve generation script | `analysis/plot_baseline_curves.py` | ✅ |
| Baseline results doc | `docs/I1_BASELINE_RESULTS.md` | ✅ |

**Scalar summaries from eval_summary.txt:**
- Eval reward peaked at step 448 (`rewards/eval/mean = 1.7113`).
- Format exact reward best at step 704 (`2.7553`).
- Final eval metrics at step 3328: `rewards/eval/mean = -0.2401`, `check_numbers = 0.1223`, `check_answer = 0.0979`, `match_format_approximately = -1.5695`, `actor/eval/kl = 0.6724`.

#### 5. Metrics

All evaluations used **greedy decoding** on the configured 64-example GSM8K test eval split (TFDS, `NUM_TEST_BATCHES=64`, `TRAIN_MICRO_BATCH_SIZE=1`, shuffle seed 42).

| Step | Model | Exact correct | Exact accuracy | Partial accuracy | Format accuracy | Empty-response count | Eval sample size | Restore confirmed | Log file |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| n/a | Base model / zero-init LoRA | 33 | 51.56% | 53.12% | 6.25% | Not counted | 64 | n/a (no restore) | `base_eval.log` |
| 2000 | LoRA-finetuned | 18 | 28.12% | 29.69% | 35.94% | Not counted | 64 | ✅ "Restored LoRA params from step 2000" | `lora_eval_step2000.log` |
| 2500 | LoRA-finetuned | 13 | 20.31% | 23.44% | 31.25% | Not counted | 64 | ✅ "Restored LoRA params from step 2500" | `lora_eval_step2500.log` |
| 3000 | LoRA-finetuned | 4 | 6.25% | 7.81% | 12.50% | Not counted | 64 | ✅ "Restored LoRA params from step 3000" | `lora_eval_step3000.log` |
| 3364 | LoRA-finetuned (final) | 2 | 3.12% | 6.25% | 12.50% | Not counted | 64 | ✅ "Restored LoRA params from step 3364" | `lora_eval.log` |

**Key observations:**
- The LoRA-finetuned model is **worse than the base model at every retained checkpoint**.
- Accuracy **degrades monotonically** from step 2000 → 3364.
- Format accuracy **improved** (6.25% → 35.94% at step 2000) but this did not translate to correctness gains.
- The degradation diagnostic notes that eval reward peaked early (steps 448–704) but those early checkpoints were not retained because `SAVE_INTERVAL_STEPS=500` and `MAX_TO_KEEP=4` — the four retained checkpoints are steps {2000, 2500, 3000, 3364}.
- **Base comparison exists on the same eval sample** (same 64 questions, same greedy decoding).
- Empty-response count was not explicitly tracked in the eval logs, but the diagnostic notes "many eval responses are empty, markdown fences, or malformed text."

#### 6. Caveats

| Caveat | Details |
|---|---|
| **Checkpoint selection** | Only 4 checkpoints retained (steps 2000–3364). Best eval reward window was steps 448–704, which were overwritten. **No best-by-eval checkpoint exists.** |
| **Small eval sample** | 64-example greedy eval is small — confidence intervals are wide (~±12 pp at 95% CI for binomial). |
| **W&B not queried** | W&B URLs recorded from local metadata but not verified. Training curves were generated from TensorBoard events. |
| **TensorBoard events not local** | The TensorBoard event file was in `/tmp/` on the TPU VM (volatile). The CSV exports in `report_assets/` are the only surviving local copies of training scalars. |
| **Checkpoint not locally available** | The preserved checkpoint is on the TPU VM at `/home/ext_felsomoye_gmail_com/tpu-2026/checkpoints/baseline_seed42/actor/3364` and as a W&B artifact. Not present in this local clone. |
| **train.log not local** | The ~19 MB training log was on the TPU VM. Not present in this local clone. |
| **Dirty worktree at launch** | Three documentation files were modified (not committed) when training launched. No training code was dirty. Minor provenance concern only. |
| **W&B entity mismatch** | `config.py` defaults to entity `milindsarkaryt-iiser-mohali` but actual W&B runs are under `felsomoye-university-of-cambridge`. The entity was presumably overridden via env var `WANDB_ENTITY`. |
| **Protobuf patch** | The only code change from the shipped baseline was `protobuf==6.31.1` in `requirements.txt`. This is dependency plumbing and does not affect model/training behaviour. |
| **TFDS/cache issues** | Two earlier attempts failed due to TFDS/protobuf incompatibility. Documented in `runs/2026-06-08_baseline_seed42_failed/`. |
| **Usability for final report** | **Usable now with caveats.** The run documents a real negative result (GRPO degradation from base). The base-model result (51.56%) is the reportable accuracy. The LoRA training demonstrates a checkpoint-selection / overfitting failure that can be reported honestly. |

---

## Hard-Question Mining

**Not used.** No hard-question manifest, mining code, or related artefacts were found in this branch. The `docs/EXPERIMENTS.md` lists only a baseline experiment and two planned variants (reward-length, KL-control). No `experiments/`, `manifests/`, or hard-mining directories exist.

---

## Missing Evidence Checklist

| # | Missing artefact | Severity | Notes |
|---:|---|---|---|
| 1 | `scripts/train.log` (~19 MB training log) | Medium | On TPU VM, not in local repo. Contains full stdout/stderr from training including per-step reward samples. |
| 2 | TensorBoard event file | Medium | Was in `/tmp/` on TPU (volatile). CSV exports exist in `report_assets/` as partial substitutes. |
| 3 | Checkpoints (local copy) | Low | Available as W&B artifact `baseline_seed42_checkpoint:latest`. Not needed unless re-evaluation is required. |
| 4 | Empty-response count per eval | Low | Not explicitly counted in eval logs; would need log parsing or re-run. |
| 5 | W&B live verification | Low | URLs recorded but W&B was not queried. |
| 6 | Early checkpoint evals (steps 448–704) | High | These were the best-performing steps according to TensorBoard eval reward, but they were not retained. **No eval data exists for them.** |
| 7 | Eval CSVs (per-question detail) | Low | Eval logs exist but per-question CSV dumps were not produced. |

---

## Report-Ready Claims

These claims are backed by artefacts present in the local repo:

1. ✅ **Base model accuracy:** Gemma-3-1b-it scores 33/64 = **51.56%** exact accuracy on the 64-example GSM8K greedy eval (log: `base_eval.log`).
2. ✅ **Baseline ran to completion:** GRPO training with K=2, seed 42, from commit `7e696c4` ran for 3364 steps in 4h 41m 38s on a v6e-1 TPU, completing with exit code 0.
3. ✅ **Final checkpoint degradation:** The final LoRA checkpoint (step 3364) scores 2/64 = **3.12%**, substantially below the base model. This is a real training/checkpoint-selection failure — retained-checkpoint evals show monotonic degradation.
4. ✅ **Best retained checkpoint:** Step 2000 scored 18/64 = 28.12%, still below base.
5. ✅ **Training curves available:** Reward and KL curves exported to `report_assets/` as CSV, PNG, and PDF.
6. ✅ **Baseline patch was dependency-only:** The protobuf pin (`protobuf==6.31.1`) did not change model, reward, data, or training behaviour.

---

## Not Yet Report-Ready

1. ⚠️ **"GRPO improves on base"** — this claim **cannot** be made. The baseline training degraded the model below its starting accuracy. A positive result would require a re-run with better checkpoint retention or early stopping.
2. ⚠️ **Variant experiments** — reward-length-penalty and KL-control variants are `planned` but not executed on this branch.
3. ⚠️ **Best-checkpoint analysis** — eval reward peaked at steps 448–704, but no checkpoints were retained from that window. A re-run with `MAX_TO_KEEP` increased or best-by-eval selection is needed to establish whether early training was actually beneficial.

---

## File created

- **Path:** `agents/evidence_funmi_20260612.md`

## Top missing artefacts

1. Early checkpoint evals (steps ~448–704) — highest priority; these represent the only window where training may have been beneficial.
2. `scripts/train.log` — full training log from TPU VM.
3. TensorBoard event file — only CSV exports survive locally.

## Can this person's runs support final report claims?

**Partially, yes.** The baseline run is well-documented and can support:
- A base-model accuracy claim (51.56%).
- An honest report of GRPO training degradation with K=2, documenting a negative result.
- A diagnostic analysis of why the run collapsed (checkpoint selection, noisy updates, KL drift).

**Cannot yet support:** A claim that GRPO training improved over the base model. This would require either a re-run with better checkpoint retention or retrieval and evaluation of early checkpoints from the W&B artifact.
