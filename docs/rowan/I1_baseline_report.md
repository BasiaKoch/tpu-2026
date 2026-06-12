# I.1 Reproducing the Baseline

Baseline GRPO run on **v6e-1**, reproduced from the shared W&B run
[`jgs4c6kl`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl)
("Baseline k=2"). The training run itself was launched from the default
`scripts/config.py` at the baseline commit; the metadata, curves, and eval below
were regenerated/verified from that run's W&B export
(`wandb_scripts/data/baseline_jgs4c6kl_*`).

## (i) Run provenance

| Item | Value | Source |
|---|---|---|
| Commit | `7e696c428687bc083604690f5f51009da6abb6d9` | `runs/2026-06-08_baseline_seed42/git_commit.txt` |
| Branch | `baseline-fls` | run record |
| Config | default `scripts/config.py` (k=2 baseline), one dependency patch (§ patches) | `config_snapshot.py` |
| Hardware | TPU v6e-1 | run record |
| Seed | 42 | `config.SEED` |
| Start (UTC) | 2026-06-08T18:17:09Z | W&B `created_at` |
| End (UTC) | 2026-06-08T22:58:47Z | W&B `heartbeat_at` |
| **Wall-clock** | **4h 41m 36s** (W&B `_runtime` = 16 896 s; run record logs 4h 41m 38s) | W&B summary |
| **GRPO steps completed** | **3364** (final logged `_step = 3364`; 3365 logged points, steps 0–3364) | W&B summary/history |
| Exit code | 0 (clean finish, `state = finished`) | W&B / `completion_summary.txt` |
| Final checkpoint | `/tmp/content/ckpts/actor/3364` | run record |
| W&B checkpoint artifact | `felsomoye-university-of-cambridge/tunix/baseline_seed42_checkpoint:latest` | run record |

`MAX_STEPS = NUM_BATCHES × NUM_ITERATIONS × TRAIN_FRACTION × NUM_EPOCHS =
3738 × 1 × 0.9 × 1 = 3364`, so the run completed the full configured schedule
(one epoch over the 90% training split, the remaining 10% held out for in-loop
eval). Key baseline hyperparameters: `RANK=ALPHA=64`, `BETA=0.08` (KL penalty),
`NUM_GENERATIONS=2` (group size G), `LEARNING_RATE=3e-6`, `TEMPERATURE=0.9`,
`MAX_GRAD_NORM=0.1`.

## (ii) GSM8K accuracy: base vs. LoRA-finetuned

**Held-out split (fixed seed).** The official GSM8K **`test`** split (TFDS,
disjoint from training), shuffled with `seed=42` (`scripts/data.py`), micro-batch
size 1, first `NUM_TEST_BATCHES=64` examples. Decoding: **greedy** preset
(`temperature=None, top_k=1, top_p=None`). Identical protocol for both models via
`scripts/evaluate.py`. Accuracy = exact numeric match of the extracted answer.

| Model | Checkpoint | Correct | **Accuracy** | Partial (±10%) | Format |
|---|---|---:|---:|---:|---:|
| (a) Base `gemma-3-1b-it` (no LoRA) | — | 33/64 | **51.56%** | 53.12% | 6.25% |
| (b) LoRA-finetuned baseline | step 3364 | 2/64 | **3.12%** | 6.25% | 12.50% |

Raw logs: `runs/2026-06-08_baseline_seed42/base_eval.log`,
`runs/2026-06-08_baseline_seed42/lora_eval.log`.

> Eval was **not** re-run during this reproduction pass — the v6e-1 was occupied
> by an active training run, and `evaluate.py` cannot share the TPU
> (libtpu single-process lock). The numbers above are the recorded results from
> the same run `jgs4c6kl` / same checkpoint artifact, and are reported verbatim.

**Result.** The final LoRA checkpoint is *substantially worse* than the base
model (3.12% vs. 51.56%). This is a genuine training/checkpoint-selection
failure, not a restore bug: the checkpoint restores successfully, and the
retained-checkpoint sweep shows monotone degradation rather than a single bad
final copy —

| Checkpoint | Accuracy |
|---|---:|
| base / no restore | 51.56% |
| step 2000 | 28.12% |
| step 2500 | 20.31% |
| step 3000 | 6.25% |
| step 3364 (final) | 3.12% |

This mirrors the training curves below: reward peaks early (~steps 400–900) then
declines, so the *final* step is far from the *best* policy. (Diagnostic:
`runs/2026-06-08_baseline_seed42/degradation_diagnostic.txt`.)

## (iii) Training curves

Regenerated from the W&B history export (`jgs4c6kl`) with
`analysis/plot_baseline_from_wandb.py`; final-step values match the team's
TensorBoard-derived assets exactly (reward `1.5625`, KL `0.5213`).

- **Mean reward** $\bar{r}$ vs. GRPO step — `actor`/`rewards/train/mean`:
  `report_assets/rowan_repro/baseline_mean_reward_curve.{png,pdf,csv}`
- **KL** $(\pi_\theta \parallel \pi_{\text{ref}})$ vs. step — `actor/train/kl`:
  `report_assets/rowan_repro/baseline_kl_curve.{png,pdf,csv}`

Reward rises to ~1.5 over the first ~500 steps, plateaus to ~step 900, then
drifts down toward 0/negative by the end. KL stays low (rolling mean ~0.5) for
most of training with sharp instability spikes near steps ~2500 and ~3300,
consistent with the late-training degradation seen in eval.

## Baseline patches

The baseline did **not** run as-shipped: TFDS/GSM8K loading failed with a
protobuf `FieldDescriptor.label` error (`protobuf 7.x` vs.
`tensorflow-datasets 4.9.9`). **Single accepted fix:** pin `protobuf==6.31.1`
in `requirements.txt`. This is dependency/environment plumbing only — model,
reward functions, data split, and all training hyperparameters are unchanged.
Full detail: `docs/BASELINE_PATCHES.md`.

This reproduction pass additionally required `pip install matplotlib` (plotting
only — not part of the training/eval path).
