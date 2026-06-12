# Evidence: Funmi Experiments, 2026-06-12

## Summary
Locally I found three relevant experiment runs plus one failed launch attempt.

Completed and usable now:
- `baseline_seed42` (`jgs4c6kl`) on `baseline-fls`
- `lora_rank128_alpha128_seed42` (`aoz8dtkp`) on `lora-rank128-alpha128-fls`
- `lora_rank128_alpha128_g2_seed42` (`v5cvlwkm`) on `lora-rank128-alpha128-fls`

Failed / not usable:
- initial `baseline_seed42` launch on `baseline-fls` failed before training during TFDS GSM8K construction

The strongest report-ready result is the `v5cvlwkm` run, because it has completed training, a confirmed checkpoint restore, base-vs-checkpoint evaluation on the same 64-example GSM8K test sample, and a preserved local checkpoint copy. The `aoz8dtkp` run is partially usable: training completed and the W&B final eval is saved locally, but the standalone restored-checkpoint exact accuracy is not present in the saved logs.

## Git State
- repo path: `/home/funmilooi-somoye/tpu-2026`
- current branch: `lora-rank128-alpha128-fls`
- current HEAD: `7e1a34ed12b081b5c1dde05f7c894f0759fe409a`
- status: clean
- relevant branches: `baseline-fls`, `lora-rank128-alpha128-fls`, `main`, `learning-rate-fls`, `n-generations-8-rerun`, and the corresponding `origin/*` refs
- relevant commits: `7e696c428687bc083604690f5f51009da6abb6d9` (patched baseline), `99059ce77f162a46c585a2ba1b9310aee4b55e9c` (G=8 + rank/alpha 128 run), `762bb69479a4d5a62889149dffa64fc0fdcbcb62` (G=2 + rank/alpha 128 run), `7e1a34ed12b081b5c1dde05f7c894f0759fe409a` (recorded G=2 eval results), `2143844` (failed baseline attempt)

## Experiment Register

| Run | Branch | Commit | Estimator | K | Seed | Steps | Status | Best exact | Final exact | W&B | Artefacts usable? |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `baseline_seed42` (`jgs4c6kl`) | `baseline-fls` / `tunix` | `7e696c428687bc083604690f5f51009da6abb6d9` | GRPO | 2 | 42 | 3364 | completed | 28.12% on step 2000 retained checkpoint | 3.12% on step 3364 restored checkpoint | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl | yes |
| `baseline_seed42` failed launch (`qvhm72qe`, relaunch `kcwsje77`) | `baseline-fls` | `2143844` | GRPO | 2 | 42 | 0 | failed | n/a | n/a | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/qvhm72qe; https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/kcwsje77 | no |
| `lora_rank128_alpha128_seed42` (`aoz8dtkp`) | `lora-rank128-alpha128-fls` | `99059ce77f162a46c585a2ba1b9310aee4b55e9c` | GRPO | 8 | 42 | 3364 | completed | 48.44% base-only eval; restored checkpoint exact not saved locally | n/a in saved logs | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/aoz8dtkp | partial |
| `lora_rank128_alpha128_g2_seed42` (`v5cvlwkm`) | `lora-rank128-alpha128-fls` | `762bb69479a4d5a62889149dffa64fc0fdcbcb62` | GRPO | 2 | 42 | 3364 | completed | 53.12% base-only eval | 32.81% restored checkpoint exact | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/v5cvlwkm | yes |

## Run Details

### Rank =128, Alpha =128, K=2 and Rank =128, Alpha =128, K=8 Experiment

#### `lora_rank128_alpha128_seed42` (`aoz8dtkp`)
- Identity
  - run nickname: `lora_rank128_alpha128_seed42`
  - run id: `aoz8dtkp`
  - owner/name: `felsomoye-university-of-cambridge/tunix/runs/aoz8dtkp`
  - team/fork: `BasiaKoch/tpu-2026`
  - branch: `lora-rank128-alpha128-fls`
  - commit hash: `99059ce77f162a46c585a2ba1b9310aee4b55e9c`
  - dirty worktree status at launch: `M scripts/config.py`
  - W&B URL/project/entity: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/aoz8dtkp`; project `tunix`; entity `felsomoye-university-of-cambridge`
  - local run root path: `/home/funmilooi-somoye/tpu-2026/runs/2026-06-10_lora_rank128_alpha128_seed42`
  - date/time launched: `2026-06-10T18:44:36Z` start recorded; W&B `startedAt` `2026-06-10T18:44:35.909108Z`
- Configuration
  - estimator: GRPO
  - `NUM_GENERATIONS`: 8
  - seed: 42
  - max steps: 3364
  - save interval: 500 steps
  - max checkpoints kept: 4
  - dataset/source: GSM8K via TFDS (`DATA_SOURCE=tfds`)
  - hard-question manifest path: not used; no local manifest found
  - reward function changes: none found in local code history for this run; the run changed LoRA capacity and group size, not reward code
  - decoding/eval preset: greedy for saved standalone eval; training-time W&B eval used the run's configured eval path
  - important env vars: `WANDB_NAME=lora_rank128_alpha128_seed42`; `WANDB_ENTITY=felsomoye-university-of-cambridge`; `WANDB_PROJECT=tunix`; `WANDB_RUN_ID` was unset in the recorded launcher command; `TPU_CONTENT_DIR` was not overridden in the recorded command
- Runtime and platform
  - TPU type/zone: local W&B metadata records `tpu.name=v6e`, `count=1`, `hbmGib=32`; repo setup notes record TPU zone `us-east5-a` and accelerator `v6e-1`
  - start/end time: `2026-06-10T18:44:36Z` to `2026-06-11T03:16:55Z`
  - wall-clock duration: `8h 32m 19s`
  - run result: completed successfully
  - failure reason: none
- Artefacts
  - run metadata: no `run_metadata.json` found locally; closest equivalents are `runs/2026-06-10_lora_rank128_alpha128_seed42/completion_summary.txt`, `runs/2026-06-10_lora_rank128_alpha128_seed42/notes.md`, and `scripts/wandb/run-20260610_184435-aoz8dtkp/files/wandb-metadata.json`
  - training log path: `/home/funmilooi-somoye/tpu-2026/scripts/train.log`
  - TensorBoard event path: `/tmp/content/tmp/tensorboard/grpo`
  - checkpoint paths and retained steps: `/home/funmilooi-somoye/tpu-2026/checkpoints/lora_rank128_alpha128_seed42/actor/` with retained steps `2000`, `2500`, `3000`, `3364`
  - eval CSV paths: no CSV eval artefacts found locally
  - eval logs: `runs/2026-06-10_lora_rank128_alpha128_seed42/base_eval.log`, `runs/2026-06-10_lora_rank128_alpha128_seed42/lora_eval.log`
  - scalar summaries: `scripts/wandb/run-20260610_184435-aoz8dtkp/files/wandb-summary.json`
  - hard-mining manifest/code: none found locally
- Metrics
  - evaluated checkpoints: base / zero-init LoRA wrapper exact `31/64 = 48.44%`
  - partial accuracy: base / zero-init LoRA wrapper `51.56%`
  - format accuracy: base / zero-init LoRA wrapper `6.25%`
  - empty-response count: not logged explicitly in local artefacts
  - eval sample size: `64`
  - checkpoint restore confirmed: not confirmed in the saved standalone `lora_eval.log`; training-time W&B eval did run, but the saved local standalone restore/eval attempt did not complete
  - base comparison exists on same eval sample: yes, via `base_eval.log`
- Caveats
  - the saved run changes both LoRA capacity (`RANK=128`, `ALPHA=128.0`) and group size (`NUM_GENERATIONS=8`), so it is not a clean G-only ablation
  - the local saved logs do not contain the standalone restored-checkpoint exact accuracy for the final checkpoint
  - W&B was not queried; the run URL and metrics were reconstructed from local logs/metadata
  - the run is usable for training-time W&B metrics and base comparison, but not yet for a final standalone checkpoint exact-accuracy claim

#### `lora_rank128_alpha128_g2_seed42` (`v5cvlwkm`)
- Identity
  - run nickname: `lora_rank128_alpha128_g2_seed42`
  - run id: `v5cvlwkm`
  - owner/name: `felsomoye-university-of-cambridge/tunix/runs/v5cvlwkm`
  - team/fork: `BasiaKoch/tpu-2026`
  - branch: `lora-rank128-alpha128-fls`
  - commit hash: `762bb69479a4d5a62889149dffa64fc0fdcbcb62`
  - dirty worktree status at launch: `M config.py` and an untracked `../runs/2026-06-11_lora_rank128_alpha128_g2_seed42/` path in the recorded pre-run status snapshot
  - W&B URL/project/entity: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/v5cvlwkm`; project `tunix`; entity `felsomoye-university-of-cambridge`
  - local run root path: `/home/funmilooi-somoye/tpu-2026/runs/2026-06-11_lora_rank128_alpha128_g2_seed42`
  - date/time launched: `2026-06-11T11:14:00Z` start recorded; W&B `startedAt` `2026-06-11T11:14:05.428011Z`
- Configuration
  - estimator: GRPO
  - `NUM_GENERATIONS`: 2
  - seed: 42
  - max steps: 3364
  - save interval: 500 steps
  - max checkpoints kept: 4
  - dataset/source: GSM8K via TFDS (`DATA_SOURCE=tfds`)
  - hard-question manifest path: not used; no local manifest found
  - reward function changes: none found in local code history for this run; this is the clean G=2 + rank/alpha 128 variant
  - decoding/eval preset: greedy for the standalone evaluation log
  - important env vars: `WANDB_NAME=lora_rank128_alpha128_g2_seed42`; `WANDB_ENTITY=felsomoye-university-of-cambridge`; `WANDB_PROJECT=tunix`; `WANDB_RUN_ID` was unset in the recorded launcher command; `TPU_CONTENT_DIR` was not overridden in the recorded command
- Runtime and platform
  - TPU type/zone: local W&B metadata records `tpu.name=v6e`, `count=1`, `hbmGib=32`; repo setup notes record TPU zone `us-east5-a` and accelerator `v6e-1`
  - start/end time: `2026-06-11T11:14:00Z` to `2026-06-11T16:43:09Z`
  - wall-clock duration: `5h 29m 9s`
  - run result: completed successfully
  - failure reason: none
- Artefacts
  - run metadata: no `run_metadata.json` found locally; closest equivalents are `runs/2026-06-11_lora_rank128_alpha128_g2_seed42/completion_summary.txt`, `runs/2026-06-11_lora_rank128_alpha128_g2_seed42/notes.md`, and `scripts/wandb/run-20260611_111405-v5cvlwkm/files/wandb-metadata.json`
  - training log path: `/home/funmilooi-somoye/tpu-2026/scripts/train.log`
  - TensorBoard event path: `/tmp/content/tmp/tensorboard/grpo`
  - checkpoint paths and retained steps: `/home/funmilooi-somoye/tpu-2026/checkpoints/lora_rank128_alpha128_g2_seed42/actor/` with retained steps `2000`, `2500`, `3000`, `3364`
  - eval CSV paths: no CSV eval artefacts found locally
  - eval logs: `runs/2026-06-11_lora_rank128_alpha128_g2_seed42/base_eval.log`, `runs/2026-06-11_lora_rank128_alpha128_g2_seed42/lora_eval.log`
  - scalar summaries: `scripts/wandb/run-20260611_111405-v5cvlwkm/files/wandb-summary.json`
  - hard-mining manifest/code: none found locally
- Metrics
  - evaluated checkpoints: base / zero-init LoRA wrapper exact `34/64 = 53.12%`
  - restored checkpoint exact accuracy: `21/64 = 32.81%`
  - partial accuracy: base / zero-init LoRA wrapper `54.69%`; restored checkpoint `32.81%`
  - format accuracy: base / zero-init LoRA wrapper `7.81%`; restored checkpoint `53.12%`
  - empty-response count: not logged explicitly in local artefacts
  - eval sample size: `64`
  - checkpoint restore confirmed: yes, `Restored LoRA params from step 3364`
  - base comparison exists on same eval sample: yes, via `base_eval.log`
- Caveats
  - the run is cleanly reportable for the final restored checkpoint exact accuracy
  - empty-response counts are still not logged explicitly
  - W&B was not queried; the run URL and metrics were reconstructed from local logs/metadata

## Hard-Question Mining
No hard-question mining evidence was found locally.

- mining method: not found locally
- source baseline: not found locally
- manifest path: not found locally
- number of questions: not found locally
- criteria for “hard”: not found locally
- branch/commit containing code: no hard-mining code path found locally
- runs using the manifest: none found locally

## Missing Evidence Checklist
- `run_metadata.json` files were not found locally for the completed runs
- empty-response counts were not logged explicitly in the saved local artefacts
- the standalone restored-checkpoint exact accuracy for `aoz8dtkp` is not present in the saved local logs
- no local hard-question mining manifest or mining output was found
- the separate TPU run root `/home/funmilooi-somoye/tpu-runs/part-i/` does not exist locally in this workspace
- I did not query W&B directly; all W&B references here come from local logs/metadata

## Report-Ready Claims
- `baseline_seed42` (`jgs4c6kl`) completed 3364 GRPO steps in `4h 41m 38s`, and the final restored checkpoint scored `2/64 = 3.12%` exact on the recorded 64-example greedy GSM8K eval
- `baseline_seed42` retained checkpoint accuracy degraded over time: step `2000` `18/64 = 28.12%`, step `2500` `13/64 = 20.31%`, step `3000` `4/64 = 6.25%`
- `lora_rank128_alpha128_seed42` (`aoz8dtkp`) completed 3364 steps in `8h 32m 19s`, and the W&B final eval metrics were `rewards/eval/mean=1.575096`, `rewards/eval/score/mean=6.300384`, `rewards/eval/check_answer=0.773981`, `actor/eval/kl=0.512891`
- `lora_rank128_alpha128_seed42` base-only greedy eval on the same 64-example GSM8K test sample was `31/64 = 48.44%` exact
- `lora_rank128_alpha128_g2_seed42` (`v5cvlwkm`) completed 3364 steps in `5h 29m 9s`, and the restored checkpoint scored `21/64 = 32.81%` exact with `53.12%` format on the same 64-example GSM8K test sample
- `lora_rank128_alpha128_g2_seed42` base-only greedy eval on the same 64-example GSM8K test sample was `34/64 = 53.12%` exact
- all three runs were launched from the TPU VM `v6e` host `t1v-n-4fcf255d-w-0` and used W&B project/entity `tunix` / `felsomoye-university-of-cambridge`

## Not Yet Report-Ready
- exact restored-checkpoint accuracy for `lora_rank128_alpha128_seed42` (`aoz8dtkp`)
- any claim that `lora_rank128_alpha128_seed42` is a clean G=8-only ablation
- empty-response counts for any checkpoint
- any hard-question mining claim
- any claim that the failed baseline launch produced a usable model checkpoint
