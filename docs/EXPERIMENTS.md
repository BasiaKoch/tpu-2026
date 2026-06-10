# Experiments

Pre-register each experiment before launching a full TPU run.

## Experiment Summary

| Experiment | Main change | Hypothesis | Success metric | Diagnostics | Status |
|---|---|---|---|---|---|
| Baseline GRPO | Default `scripts/config.py` and rewards (`NUM_GENERATIONS = 2`). | Baseline should reproduce the expected reward/KL behaviour and improve over the base model. | GSM8K eval accuracy. | Reward curve, KL curve, output format accuracy. | completed (run `jgs4c6kl`) |
| Group size G=8 | `NUM_GENERATIONS` 2 → 8 in `scripts/config.py`. | A larger GRPO group gives a less noisy advantage baseline/normalisation, so policy updates are more stable and avoid the late-training collapse seen in the baseline. | Eval reward mean (and GSM8K accuracy) vs. baseline. | Eval/train reward curves, KL, grad-norm spikes, completion length, empty-completion rate. | running |
| Micro batch size 2 | `TRAIN_MICRO_BATCH_SIZE` 1 → 2 in `scripts/config.py`. | A larger micro-batch should improve TPU compute efficiency by doing more useful work per step, as long as memory headroom is sufficient and training dynamics remain comparable. | Wall-clock time per step and eval reward/GSM8K accuracy vs. baseline. | TPU utilisation, step time, OOMs, KL, grad norm, reward curves. | planned |
| Variant A: reward/length | Add a mild length penalty or adjust shaping reward. | Reducing verbosity or reward hacking may improve correctness without excessive KL drift. | GSM8K eval accuracy vs. baseline. | Response length, malformed-output rate, correctness reward. | planned |
| Variant A on G8: reward/length + G8 | `length_penalty` reward on top of `n-generations-8` (`NUM_GENERATIONS = 8` **and** `TRAIN_MICRO_BATCH_SIZE = 2`). | On the more stable G8 base, a mild excess-length penalty should further curb length-blowup / verbosity reward-hacking without adding KL drift. | Eval reward mean & GSM8K accuracy vs. the **G8 run** (not baseline). | Completion length, empty-completion rate, KL, grad-norm, correctness reward. | planned |
| Variant B: KL control | Conservative change to beta or epsilon. | KL budget should trade off stability against policy movement. | Accuracy read alongside KL. | KL vs. reward and KL vs. accuracy. | planned |

## Logged Experiments

### `group-size-g8`

**Question:** Does increasing the GRPO group size `NUM_GENERATIONS` from 2 to 8 fix the unstable, collapsing training seen in the baseline?

**Main change:** `NUM_GENERATIONS = 2 → 8` in [scripts/config.py:44](../scripts/config.py). All other hyperparameters unchanged.

**Hypothesis:** At G=2 the per-prompt advantage baseline is a 2-sample mean and the std-normalisation is extremely noisy, producing high-variance policy gradients. This is the likely driver of the baseline's late-training collapse (KL spike to 41, grad-norm spike to 696, completions degenerating to 0 length — see [baseline-analysis.md](rowan/baseline-analysis.md)). G=8 should give a much lower-variance advantage estimate, keeping KL and grad-norm controlled and sustaining the early reward gains instead of collapsing.

**Config:** Branch TBD; commit TBD (the `2 → 8` edit). Seed: not yet fixed. Data: GSM8K, `TRAIN_FRACTION = 0.9`. Steps: `MAX_STEPS = 3364`. Changed parameter: `NUM_GENERATIONS = 8`.

**Comparison:** Against baseline run `jgs4c6kl` (G=2) — see [RUNS.md](RUNS.md).

**Metrics:** Eval reward mean, `rewards/eval/check_answer`, KL (`actor/train/kl`), grad-norm, completion length, empty-completion rate, format-match rewards.

**Result:** _Pending — fill in after the run completes._

**Interpretation:** _Pending._

Baseline reference (G=2, `jgs4c6kl`): peak eval reward mean **1.711** at step **448**, collapsing to **-0.240** by the final eval (step 3328); wall-clock ≈ **4.7 h** over 3,364 steps.

### `micro-batch-size-2`

**Question:** Does increasing `TRAIN_MICRO_BATCH_SIZE` from 1 to 2 improve TPU compute efficiency without hurting training quality?

**Main change:** `TRAIN_MICRO_BATCH_SIZE = 1 → 2` in [scripts/config.py:52](../scripts/config.py). All other hyperparameters should stay unchanged for the controlled comparison.

**Hypothesis:** With a larger micro-batch, each training step should do more useful work per compile/execution overhead and improve TPU utilisation, reducing wall-clock time per effective amount of training. The main risk is insufficient memory headroom or a change in optimisation behaviour from the larger per-step batch.

**Config:** Branch TBD; commit TBD. Seed: not yet fixed. Data: GSM8K, `TRAIN_FRACTION = 0.9`. Steps: `MAX_STEPS = 3364`. Changed parameter: `TRAIN_MICRO_BATCH_SIZE = 2`.

**Comparison:** Against baseline run `jgs4c6kl` (`TRAIN_MICRO_BATCH_SIZE = 1`) — see [RUNS.md](RUNS.md). Compare both compute metrics and model-quality metrics, not step time alone.

**Metrics:** Wall-clock time per step, total runtime, TPU utilisation if available, eval reward mean, `rewards/eval/check_answer`, GSM8K accuracy, KL (`actor/train/kl`), grad-norm, OOM/retry events.

**Result:** _Pending — fill in after the run completes._

**Interpretation:** _Pending._

### `reward-length-on-g8`

**Question:** Conditional on the G8 base, does adding a mild excess-length penalty further reduce verbosity / length-blowup reward-hacking and improve GSM8K accuracy without added KL drift?

**Main change:** Add the `length_penalty` reward fn to [scripts/rewards.py](../scripts/rewards.py) and append it to `REWARD_FNS`, on top of the `n-generations-8` (G8) branch. One `config.py` change vs that branch: `TRAIN_MICRO_BATCH_SIZE` reduced **2 → 1**, because G8 + micro-batch-2 exhausts HBM on the single-chip v6e-1 (see deviation note). The control run uses the same bs=1, so the length penalty stays the only delta between control and treatment.

**Base / deviation note:** Relative to the original baseline this stacks several changes (G8, `TRAIN_MICRO_BATCH_SIZE` reduced to 1, *and* the length penalty), so it does **not** isolate the length penalty against the baseline. It deliberately deviates from the original `reward-length-bk` pre-registration, which held group size `K` fixed. **Hardware note:** this runs on a single-chip **v6e-1**, where Rowan's G8 + micro-batch-2 exhausts HBM after ~5h (a long-completion batch needs ~28 GB vs ~25 GB free → `RESOURCE_EXHAUSTED`), so micro-batch is reduced **2 → 1** on *both* arms. The declared control is therefore a **G8 + bs=1** run (`n-generations-8` with the same micro-batch edit), against which the only delta is the `length_penalty` reward. Because of the bs change, results are **not** directly comparable to Rowan's bs=2 G8 run.

**Hypothesis:** G8 already lowers advantage-estimate variance and should curb the late-training collapse; the length penalty addresses a different failure mode (reward-hacking via verbose completions). On top of a more stable base, the penalty should pull mean completion length down and reduce empty/degenerate completions without raising KL, yielding equal-or-better GSM8K accuracy than G8 alone.

**Config:** Branch `reward-length-on-g8-bk` (off `origin/n-generations-8`); commit: this branch HEAD. `NUM_GENERATIONS = 8`, `TRAIN_MICRO_BATCH_SIZE = 1`. Data: GSM8K, `TRAIN_FRACTION = 0.9`. Steps: `MAX_STEPS = 3364`. Checkpoints on persistent disk via `TPU_CONTENT_DIR=/home/basiakoch/content`. Delta vs the G8+bs1 control: `REWARD_FNS` gains `length_penalty` (`LENGTH_TARGET = 600`, `LENGTH_PENALTY_WEIGHT = 1.0`).

**Determinism / effective seeds (verified against tunix + qwix source):** three randomness sources, all deterministic and identical to the G8 control:
- Data shuffle — `seed=42`, explicit in [scripts/data.py:86](../scripts/data.py).
- Rollout sampling — `RolloutConfig.seed` is left unset, so tunix's sampler falls back to `jax.random.PRNGKey(0)` (`tunix/generate/sampler.py`); deterministic.
- LoRA init — `model.py` passes no `rngs`, so qwix falls back to `nnx.Rngs(10003)` (`qwix/_src/providers/lora.py`; `lora_a = he_uniform`, `lora_b = zeros`); deterministic.

Seeds 2 and 3 are **library defaults, not pinned in this repo**, and tunix/qwix are installed from unpinned GitHub HEAD (`bootstrap.sh`). The comparison is valid only if this run and the G8 control share the same venv/library revisions. Do **not** set the rollout/LoRA seeds explicitly on this branch — that would diverge the RNG stream from the G8 control and reintroduce a confound.

**Comparison:** Against a **G8 + bs=1 control** run on the same v6e-1 VM (`n-generations-8` with `TRAIN_MICRO_BATCH_SIZE = 1`) — *not* baseline `jgs4c6kl`, and *not* Rowan's bs=2 G8 run. Both arms run back-to-back on the same VM/venv so hardware and library revisions match by construction. Uninterpretable until the control run completes; a secondary read against the isolated length-only arm (`reward-length-bk`) would reveal any G8×length interaction.

**Metrics:** Completion length vs step (primary), empty-completion rate, eval reward mean, `rewards/eval/check_answer`, GSM8K accuracy across checkpoints, KL (`actor/train/kl`), grad-norm.

**Risk:** If the penalty is too strong it may suppress reasoning or push toward short/empty answers — especially relevant since G8 already shifts completion-length dynamics.

**Result:** _Pending._

**Interpretation:** _Pending._

## Template

### `<experiment-name>`

**Question:** What does this experiment test?

**Main change:** One controlled code/config change.

**Hypothesis:** What should happen, and why?

**Config:** Branch, commit, seed, data split, total steps, changed parameters.

**Comparison:** Which baseline/run is this compared against?

**Metrics:** Accuracy, reward, KL, response length, format accuracy, or other diagnostics.

**Result:** Fill in after evaluation.

**Interpretation:** Did the outcome match the hypothesis?
