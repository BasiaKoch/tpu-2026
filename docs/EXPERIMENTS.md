# Experiments

Pre-register each experiment before launching a full TPU run.

## Experiment Summary

| Experiment | Main change | Hypothesis | Success metric | Diagnostics | Status |
|---|---|---|---|---|---|
| Baseline GRPO | Default `scripts/config.py` and rewards (`NUM_GENERATIONS = 2`). | Baseline should reproduce the expected reward/KL behaviour and improve over the base model. | GSM8K eval accuracy. | Reward curve, KL curve, output format accuracy. | completed (run `jgs4c6kl`); LoRA eval below base on 64-example greedy eval |
| Group size G=8 | `NUM_GENERATIONS` 2 → 8 in `scripts/config.py`. | A larger GRPO group gives a less noisy advantage baseline/normalisation, so policy updates are more stable and avoid the late-training collapse seen in the baseline. | Eval reward mean (and GSM8K accuracy) vs. baseline. | Eval/train reward curves, KL, grad-norm spikes, completion length, empty-completion rate. | superseded/invalid as clean ablation: run `aoz8dtkp` also changed `RANK` and `ALPHA` to 128 |
| LoRA rank/alpha 128 + G=8 | `RANK=128`, `ALPHA=128.0`, `NUM_GENERATIONS=8`. | More adapter capacity plus larger GRPO group should improve stability relative to the collapsed G=2 final checkpoint. | Eval reward mean and standalone GSM8K accuracy vs. baseline. | Eval/train reward curves, KL, grad-norm, completion length, standalone restored-checkpoint eval. | completed training (`aoz8dtkp`); standalone LoRA accuracy pending |
| Micro batch size 2 | `TRAIN_MICRO_BATCH_SIZE` 1 → 2 in `scripts/config.py`. | A larger micro-batch should improve TPU compute efficiency by doing more useful work per step, as long as memory headroom is sufficient and training dynamics remain comparable. | Wall-clock time per step and eval reward/GSM8K accuracy vs. baseline. | TPU utilisation, step time, OOMs, KL, grad norm, reward curves. | planned |
| Variant A: reward/length | Add a mild length penalty or adjust shaping reward. | Reducing verbosity or reward hacking may improve correctness without excessive KL drift. | GSM8K eval accuracy vs. baseline. | Response length, malformed-output rate, correctness reward. | planned |
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

Note: run `aoz8dtkp` completed with `NUM_GENERATIONS=8`, but it also changed `RANK=128` and `ALPHA=128.0`, so it should not be used as the clean result for this experiment.

### `lora-rank128-alpha128-g8`

**Question:** Does increasing LoRA rank/alpha to 128 while using `NUM_GENERATIONS=8` avoid the final collapse seen in the baseline?

**Main change:** `RANK=64 → 128`, `ALPHA=64.0 → 128.0`, and `NUM_GENERATIONS=2 → 8`.

**Hypothesis:** Higher adapter capacity plus lower-variance GRPO group normalisation should allow useful policy movement without the severe final-step collapse seen in the baseline.

**Config:** Commit `99059ce77f162a46c585a2ba1b9310aee4b55e9c`; seed `42`; GSM8K with `TRAIN_FRACTION=0.9`; `MAX_STEPS=3364`; W&B run `aoz8dtkp`.

**Comparison:** Baseline run `jgs4c6kl` final checkpoint and early best checkpoint. Treat as a combined-parameter variant, not a single-variable ablation.

**Metrics:** W&B final eval reward, answer reward, KL, grad norm, completion length, and standalone GSM8K accuracy from the restored checkpoint.

**Result:** Training completed in 8h 32m 19s. Final W&B eval: `rewards/eval/mean=1.575096`, `rewards/eval/score/mean=6.300384`, `rewards/eval/check_answer=0.773981`, `actor/eval/kl=0.512891`, `completions/eval/mean_length=460.006`. Standalone base eval was `34/64` (`53.12%`); standalone restored-checkpoint accuracy was `21/64` (`32.81%`).

**Interpretation:** The W&B final eval does not show the obvious collapse seen in the G=2 baseline final checkpoint, but the standalone restored checkpoint still underperforms the base model on GSM8K greedy accuracy.

### `micro-batch-size-2`

**Question:** Does increasing `TRAIN_MICRO_BATCH_SIZE` from 1 to 2 improve TPU compute efficiency without hurting training quality?

**Main change:** `TRAIN_MICRO_BATCH_SIZE = 1 → 2` in [scripts/config.py:52](../scripts/config.py). All other hyperparameters should stay unchanged for the controlled comparison.

**Hypothesis:** With a larger micro-batch, each training step should do more useful work per compile/execution overhead and improve TPU utilisation, reducing wall-clock time per effective amount of training. The main risk is insufficient memory headroom or a change in optimisation behaviour from the larger per-step batch.

**Config:** Branch TBD; commit TBD. Seed: not yet fixed. Data: GSM8K, `TRAIN_FRACTION = 0.9`. Steps: `MAX_STEPS = 3364`. Changed parameter: `TRAIN_MICRO_BATCH_SIZE = 2`.

**Comparison:** Against baseline run `jgs4c6kl` (`TRAIN_MICRO_BATCH_SIZE = 1`) — see [RUNS.md](RUNS.md). Compare both compute metrics and model-quality metrics, not step time alone.

**Metrics:** Wall-clock time per step, total runtime, TPU utilisation if available, eval reward mean, `rewards/eval/check_answer`, GSM8K accuracy, KL (`actor/train/kl`), grad-norm, OOM/retry events.

**Result:** _Pending — fill in after the run completes._

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
