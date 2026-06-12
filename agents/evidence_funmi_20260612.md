# Evidence: Funmi Experiments, 2026-06-12

## Summary
Four experiment records are supported locally:
- `baseline_seed42` is fully reportable.
- `lr1e5_seed42` is reportable, but its archived config snapshot disagrees with the run note on `K`, so the `K` value is provenance-uncertain.
- `lora_rank128_alpha128_seed42` is only partially reportable right now because the committed run artifact does not include a completed standalone restored-checkpoint exact-accuracy log in the saved evidence.
- `lora_rank128_alpha128_g2_seed42` is fully reportable.

No local `~/tpu-runs/part-i/` root was found. W&B was not queried; all W&B URLs below come from logs, run folders, or committed run metadata.

## Git State
- repo path: `/home/funmilooi-somoye/tpu-2026`
- current branch: `learning-rate-fls`
- current HEAD: `0f437b91f7a53991257163df308f2c4b3eb7b19f`
- status: clean before creating this evidence file; the only new local change from this audit is `agents/evidence_funmi_20260612.md`
- relevant branches: `baseline-fls`, `learning-rate-fls`, `lora-rank128-alpha128-fls`, `main`, `n-generations-8-rerun`
- relevant commits: `7e696c428687bc083604690f5f51009da6abb6d9`, `99059ce77f162a46c585a2ba1b9310aee4b55e9c`, `2cd43abc4d3fb4e584cb9004be47b4fd7036876e`, `0f437b91f7a53991257163df308f2c4b3eb7b19f`, `762bb69479a4d5a62889149dffa64fc0fdcbcb62`, `9648612886d338884ff2cceef8885508942122c3`, `7e1a34ed12b081b5c1dde05f7c894f0759fe409a`, `1d7a542a894266ca380a42ff2317d3559dbf5266`

## Experiment Register

| Run | Branch | Commit | Estimator | K | Seed | Steps | Status | Best exact | Final exact | W&B | Artefacts usable? |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `baseline_seed42` | `baseline-fls` | `7e696c428687bc083604690f5f51009da6abb6d9` | GRPO | 2 | 42 | 3364 | completed | 28.12% | 3.12% | `jgs4c6kl` | yes |
| `lr1e5_seed42` | `learning-rate-fls` | `99059ce77f162a46c585a2ba1b9310aee4b55e9c` | GRPO | 8* | 42 | 3364 | completed | 31.25% | 29.69% | `hozux9t6` | yes |
| `lora_rank128_alpha128_seed42` | `baseline-fls / tunix` | `99059ce77f162a46c585a2ba1b9310aee4b55e9c` | GRPO | 8 | 42 | 3364 | completed | n/a | n/a | `aoz8dtkp` | no |
| `lora_rank128_alpha128_g2_seed42` | `lora-rank128-alpha128-fls` | `762bb69479a4d5a62889149dffa64fc0fdcbcb62` | GRPO | 2 | 42 | 3364 | completed | 32.81% | 32.81% | `v5cvlwkm` | yes |

`*` `lr1e5_seed42` is provenance-uncertain on `K`: the run note says only the learning rate changed, but the archived config snapshot for the run shows `NUM_GENERATIONS = 8`.

## Run Details

### Baseline GRPO

Identity
- Run nickname: `baseline_seed42`
- Run id: `jgs4c6kl`
- Owner/name: `felsomoye-university-of-cambridge/tunix`, run name `leafy-microwave-6`
- Team/fork: `tunix` on `baseline-fls`
- Branch: `baseline-fls`
- Commit hash: `7e696c428687bc083604690f5f51009da6abb6d9`
- Dirty worktree at launch: `M docs/BASELINE_PATCHES.md`, `M docs/RUNS.md`, `M docs/SETUP_NOTES.md` in the launch ledger
- W&B URL/project/entity: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl`
- Local run root path: `runs/2026-06-08_baseline_seed42/`
- Date/time launched: `2026-06-08T18:17:09.756585Z`

Configuration
- Estimator: GRPO
- `NUM_GENERATIONS`: 2
- Seed: 42
- Max steps: 3364
- Save interval: 500
- Max checkpoints kept: 4
- Dataset/source: GSM8K via TFDS
- Hard-question manifest: not found locally
- Reward function changes: none beyond the baseline reward file
- Decoding/eval preset: `greedy`
- Important env vars: `WANDB_PROJECT=tunix`, `WANDB_ENTITY=felsomoye-university-of-cambridge`, `DATA_SOURCE=tfds`

Runtime and platform
- TPU: `v6e-1`
- Start/end: `2026-06-08T18:17:09Z` to `2026-06-08T22:58:47Z`
- Wall-clock: `4h 41m 38s`
- Status: completed
- Failure reason: none

Artefacts
- `run_metadata.json`: not found locally
- Training log: `scripts/train.log`
- TensorBoard event path: configured logdir `/tmp/content/tmp/tensorboard/grpo`; event file not found locally
- Checkpoints: `/tmp/content/ckpts/actor/2000`, `/2500`, `/3000`, `/3364`; preserved copy at `checkpoints/baseline_seed42/actor/3364`
- Eval logs: `runs/2026-06-08_baseline_seed42/base_eval.log`, `runs/2026-06-08_baseline_seed42/lora_eval.log`, `runs/2026-06-08_baseline_seed42/lora_eval_step2000.log`, `runs/2026-06-08_baseline_seed42/lora_eval_step2500.log`, `runs/2026-06-08_baseline_seed42/lora_eval_step3000.log`
- Scalar summaries: `runs/2026-06-08_baseline_seed42/eval_summary.txt`, W&B summary in `scripts/wandb/run-20260609_211839-hozux9t6/files/wandb-summary.json`
- Hard-mining manifest/code: not found locally

Metrics
- Base comparison on same eval sample: yes, `33/64` exact, `53.12%` partial, `6.25%` format
- Restored checkpoint restore confirmed: yes, `Restored LoRA params from step 3364`
- Step 2000: `18/64` exact, `28.12%` partial, `35.94%` format
- Step 2500: `13/64` exact, `20.31%` partial, `31.25%` format
- Step 3000: `4/64` exact, `6.25%` partial, `12.50%` format
- Step 3364: `2/64` exact, `6.25%` partial, `12.50%` format
- Empty-response count: not recorded in the saved eval logs

Caveats
- The final checkpoint is usable for the final report.
- This run is a strong negative result: the checkpoint degraded sharply relative to the base comparison.

### Learning Rate = 1e-5

Identity
- Run nickname: `lr1e5_seed42`
- Run id: `hozux9t6`
- Owner/name: `felsomoye-university-of-cambridge/tunix`, run name `wobbly-gorge-22`
- Team/fork: `tunix` on `learning-rate-fls`
- Branch: `learning-rate-fls`
- Commit hash: `99059ce77f162a46c585a2ba1b9310aee4b55e9c`
- Dirty worktree at launch: `M scripts/config.py`
- W&B URL/project/entity: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/hozux9t6`
- Local run root path: `runs/2026-06-09_lr1e5_seed42/`
- Date/time launched: `2026-06-09T21:18:35Z`

Configuration
- Estimator: GRPO
- `NUM_GENERATIONS`: `8` in the archived config snapshot; the run note says only the learning rate changed relative to baseline, so this field is provenance-uncertain
- Seed: 42
- Max steps: 3364
- Save interval: 500
- Max checkpoints kept: 4
- Dataset/source: GSM8K via TFDS
- Hard-question manifest: not found locally
- Reward function changes: none recorded
- Decoding/eval preset: `greedy`
- Important env vars: `WANDB_PROJECT=tunix`, `WANDB_ENTITY=felsomoye-university-of-cambridge`, `DATA_SOURCE=tfds`, `TPU_CONTENT_DIR=/tmp/content`

Runtime and platform
- TPU: `v6e-1`
- Start/end: `2026-06-09T21:18:35Z` to `2026-06-10T05:21:57Z`
- Wall-clock: `8h 03m 22s`
- Status: completed
- Failure reason: none

Artefacts
- `run_metadata.json`: not found locally
- Training log: `runs/2026-06-09_lr1e5_seed42/live_log.txt`
- TensorBoard event path: configured logdir `/tmp/content/tmp/tensorboard/grpo`; event file not found locally
- Checkpoints: `/tmp/content/ckpts/actor/2000`, `/2500`, `/3000`, `/3364`; preserved copy at `checkpoints/lr1e5_seed42/actor/3364`
- Eval logs: `runs/2026-06-09_lr1e5_seed42/base_eval.log`, `runs/2026-06-09_lr1e5_seed42/lora_eval.log`, `runs/2026-06-09_lr1e5_seed42/lora_eval_step2000.log`, `runs/2026-06-09_lr1e5_seed42/lora_eval_step2500.log`, `runs/2026-06-09_lr1e5_seed42/lora_eval_step3000.log`
- Scalar summaries: `runs/2026-06-09_lr1e5_seed42/eval_summary.txt`, `runs/2026-06-09_lr1e5_seed42/wandb_export/wandb_hozux9t6_summary.json`
- Hard-mining manifest/code: not found locally

Metrics
- Base comparison on same eval sample: yes, `33/64` exact, `53.12%` partial, `6.25%` format
- Restored checkpoint restore confirmed: yes, `Restored LoRA params from step 3364`
- Step 2000: `20/64` exact, `31.25%` partial, `98.44%` format
- Step 2500: `20/64` exact, `31.25%` partial, `95.31%` format
- Step 3000: `17/64` exact, `26.56%` partial, `96.88%` format
- Step 3364: `19/64` exact, `34.38%` partial, `98.44%` format
- Empty-response count: not recorded in the saved eval logs

Caveats
- The run is usable for the final report, but the archived config snapshot does not match the run note on `K`, so treat the `K=8` field as provenance-uncertain.
- This run improves format adherence but not base-model exact accuracy.

### LoRA Rank/Alpha 128, K=8

Identity
- Run nickname: `lora_rank128_alpha128_seed42`
- Run id: `aoz8dtkp`
- Owner/name: `felsomoye-university-of-cambridge/tunix`, run name `lora_rank128_alpha128_seed42`
- Team/fork: `tunix` on `baseline-fls`
- Branch: `baseline-fls`
- Commit hash: `99059ce77f162a46c585a2ba1b9310aee4b55e9c`
- Dirty worktree at launch: `M scripts/config.py`
- W&B URL/project/entity: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/aoz8dtkp`
- Local run root path: `runs/2026-06-10_lora_rank128_alpha128_seed42/` in git history; not present in the current checkout
- Date/time launched: `2026-06-10T18:44:36Z`

Configuration
- Estimator: GRPO
- `NUM_GENERATIONS`: 8
- Seed: 42
- Max steps: 3364
- Save interval: 500
- Max checkpoints kept: 4
- Dataset/source: GSM8K via TFDS
- Hard-question manifest: not found locally
- Reward function changes: none recorded
- Decoding/eval preset: `greedy`
- Important env vars: `WANDB_PROJECT=tunix`, `WANDB_ENTITY=felsomoye-university-of-cambridge`, `DATA_SOURCE=tfds`

Runtime and platform
- TPU: `v6e-1`
- Start/end: `2026-06-10T18:44:36Z` to `2026-06-11T03:16:55Z`
- Wall-clock: `8h 32m 19s`
- Status: completed
- Failure reason: none

Artefacts
- `run_metadata.json`: not found locally
- Training log: `scripts/train.log`
- TensorBoard event path: configured logdir `/tmp/content/tmp/tensorboard/grpo`; event file not found locally
- Checkpoints: `/tmp/content/ckpts/actor/3364`; preserved copy at `checkpoints/lora_rank128_alpha128_seed42/actor/`
- Eval logs: `runs/2026-06-10_lora_rank128_alpha128_seed42/base_eval.log`
- Scalar summaries: `runs/2026-06-10_lora_rank128_alpha128_seed42/eval_summary.txt`, `runs/2026-06-10_lora_rank128_alpha128_seed42/completion_summary.txt`
- Hard-mining manifest/code: not found locally

Metrics
- Base comparison on same eval sample: yes, `31/64` exact, `51.56%` partial, `6.25%` format
- Restored checkpoint restore confirmed: not in the saved run folder
- Final W&B training-time eval: `rewards/eval/mean=1.575096`, `rewards/eval/score/mean=6.300384`, `rewards/eval/check_answer=0.773981`, `actor/eval/kl=0.512891`, `completions/eval/mean_length=460.006`
- Standalone restored-checkpoint exact accuracy: not saved yet in the saved evidence
- Empty-response count: not recorded in the saved eval logs

Caveats
- Not usable for final report claims yet because the committed evidence does not include a completed standalone restored-checkpoint exact-accuracy result.
- This run is also not a clean G-only ablation because it changes both LoRA capacity and `K`.

### LoRA Rank/Alpha 128, K=2

Identity
- Run nickname: `lora_rank128_alpha128_g2_seed42`
- Run id: `v5cvlwkm`
- Owner/name: `felsomoye-university-of-cambridge/tunix`, run name `lora_rank128_alpha128_g2_seed42`
- Team/fork: `tunix` on `lora-rank128-alpha128-fls`
- Branch: `lora-rank128-alpha128-fls`
- Commit hash: `762bb69479a4d5a62889149dffa64fc0fdcbcb62`
- Dirty worktree at launch: `M config.py`, plus `?? ../runs/2026-06-11_lora_rank128_alpha128_g2_seed42/`
- W&B URL/project/entity: `https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/v5cvlwkm`
- Local run root path: `runs/2026-06-11_lora_rank128_alpha128_g2_seed42/` in git history; not present in the current checkout
- Date/time launched: `2026-06-11T11:14:00Z`

Configuration
- Estimator: GRPO
- `NUM_GENERATIONS`: 2
- Seed: 42
- Max steps: 3364
- Save interval: 500
- Max checkpoints kept: 4
- Dataset/source: GSM8K via TFDS
- Hard-question manifest: not found locally
- Reward function changes: none recorded
- Decoding/eval preset: `greedy`
- Important env vars: `WANDB_PROJECT=tunix`, `WANDB_ENTITY=felsomoye-university-of-cambridge`, `DATA_SOURCE=tfds`

Runtime and platform
- TPU: `v6e-1`
- Start/end: `2026-06-11T11:14:00Z` to `2026-06-11T16:43:09Z`
- Wall-clock: `5h 29m 9s`
- Status: completed
- Failure reason: none

Artefacts
- `run_metadata.json`: not found locally
- Training log: `scripts/train.log`
- TensorBoard event path: configured logdir `/tmp/content/tmp/tensorboard/grpo`; event file not found locally
- Checkpoints: `/tmp/content/ckpts/actor/3364`; preserved copy at `checkpoints/lora_rank128_alpha128_g2_seed42/actor/`
- Eval logs: `runs/2026-06-11_lora_rank128_alpha128_g2_seed42/base_eval.log`, `runs/2026-06-11_lora_rank128_alpha128_g2_seed42/lora_eval.log`
- Scalar summaries: `runs/2026-06-11_lora_rank128_alpha128_g2_seed42/eval_summary.txt`, `runs/2026-06-11_lora_rank128_alpha128_g2_seed42/completion_summary.txt`
- Hard-mining manifest/code: not found locally

Metrics
- Base comparison on same eval sample: yes, `34/64` exact, `54.69%` partial, `7.81%` format
- Restored checkpoint restore confirmed: yes, `Restored LoRA params from step 3364`
- Step 3364: `21/64` exact, `32.81%` partial, `53.12%` format
- Empty-response count: not recorded in the saved eval logs

Caveats
- This run is usable for the final report now.
- It is a combined LoRA-capacity and `K=2` variant, not a pure `K` ablation.

## Hard-Question Mining
No hard-question mining is evidenced locally.
- Mining method: not found locally
- Source baseline: not found locally
- Manifest path: not found locally
- Number of questions: not found locally
- Criteria for “hard”: not found locally
- Branch/commit containing code: not found locally
- Runs using the manifest: none found locally

## Missing Evidence Checklist
- `run_metadata.json` for each run was not found locally.
- TensorBoard `events.out.tfevents*` files were not found locally under the configured logdirs.
- Empty-response counts were not recorded in the saved eval logs.
- A clean provenance record for `lr1e5_seed42`’s `K` value is still missing because the run note and archived config snapshot disagree.
- A completed standalone restored-checkpoint exact-accuracy log for `lora_rank128_alpha128_seed42` is not present in the saved evidence.
- No local `~/tpu-runs/part-i/` root was found.
- No hard-question mining manifest or code was found locally.
- W&B server-side verification was not run; only local logs, run folders, and committed metadata were used.

## Report-Ready Claims
- `baseline_seed42` completed on `baseline-fls` at step 3364 and its final restored checkpoint scored `2/64` exact, `6.25%` partial, `12.50%` format.
- `baseline_seed42` retained checkpoints show a monotonic degradation from step 2000 to 3364.
- `lr1e5_seed42` completed on `learning-rate-fls` at step 3364 and its final restored checkpoint scored `19/64` exact, `34.38%` partial, `98.44%` format.
- `lr1e5_seed42` reached its best retained exact accuracy at `31.25%` on steps 2000 and 2500.
- `lora_rank128_alpha128_g2_seed42` completed on `lora-rank128-alpha128-fls` at step 3364 and its restored checkpoint scored `21/64` exact, `32.81%` partial, `53.12%` format.

## Not Yet Report-Ready
- `lora_rank128_alpha128_seed42` has only base-eval and training-time W&B evidence in the saved run folder; the committed evidence does not include a completed standalone restored-checkpoint exact-accuracy result.
- `lr1e5_seed42` is usable, but any claim that it is a clean LR-only ablation should be treated as provenance-uncertain because the archived config snapshot shows `NUM_GENERATIONS = 8`.
- There is no locally found clean `K=8` only or `K=2` plus `lr=1e-5` ablation record beyond the mixed-parameter runs above.
