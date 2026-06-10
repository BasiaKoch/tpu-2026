# Experiments

Pre-register each experiment before launching a full TPU run.

## Experiment Summary

| Experiment | Main change | Hypothesis | Success metric | Diagnostics | Status |
|---|---|---|---|---|---|
| Baseline GRPO | Default `scripts/config.py` and rewards (`NUM_GENERATIONS = 2`). | Baseline should reproduce the expected reward/KL behaviour and improve over the base model. | GSM8K eval accuracy. | Reward curve, KL curve, output format accuracy. | completed (run `jgs4c6kl`); LoRA eval below base on 64-example greedy eval |
| Group size G=8 | `NUM_GENERATIONS` 2 → 8 in `scripts/config.py`. | A larger GRPO group gives a less noisy advantage baseline/normalisation, so policy updates are more stable and avoid the late-training collapse seen in the baseline. | Eval reward mean (and GSM8K accuracy) vs. baseline. | Eval/train reward curves, KL, grad-norm spikes, completion length, empty-completion rate. | completed (`4i8lcitv`); CPU-fallback exact eval recovered 38/64, with W&B scalar eval reward mean 1.908 and score mean 7.633 |
| Micro batch size 2 | `TRAIN_MICRO_BATCH_SIZE` 1 → 2 in `scripts/config.py`. | A larger micro-batch should improve TPU compute efficiency by doing more useful work per step, as long as memory headroom is sufficient and training dynamics remain comparable. | Wall-clock time per step and eval reward/GSM8K accuracy vs. baseline. | TPU utilisation, step time, OOMs, KL, grad norm, reward curves. | planned |
| Variant A: reward/length | Add a mild length penalty or adjust shaping reward. | Reducing verbosity or reward hacking may improve correctness without excessive KL drift. | GSM8K eval accuracy vs. baseline. | Response length, malformed-output rate, correctness reward. | planned |
| Variant B: KL control | Conservative change to beta or epsilon. | KL budget should trade off stability against policy movement. | Accuracy read alongside KL. | KL vs. reward and KL vs. accuracy. | planned |

## Logged Experiments

### `group-size-g8`

**Question:** Does increasing the GRPO group size `NUM_GENERATIONS` from 2 to 8 fix the unstable, collapsing training seen in the baseline?

**Main change:** `NUM_GENERATIONS = 2 → 8` in [scripts/config.py:44](../scripts/config.py). All other hyperparameters unchanged.

**Hypothesis:** At G=2 the per-prompt advantage baseline is a 2-sample mean and the std-normalisation is extremely noisy, producing high-variance policy gradients. This is the likely driver of the baseline's late-training collapse (KL spike to 41, grad-norm spike to 696, completions degenerating to 0 length — see [baseline-analysis.md](rowan/baseline-analysis.md)). G=8 should give a much lower-variance advantage estimate, keeping KL and grad-norm controlled and sustaining the early reward gains instead of collapsing.

**Config:** Branch `main`; commit `99059ce77f162a46c585a2ba1b9310aee4b55e9c`. Seed: `42`. Data: GSM8K, `TRAIN_FRACTION = 0.9`. Steps: `MAX_STEPS = 3364`. Changed parameters: `NUM_GENERATIONS = 8`, `LEARNING_RATE = 3e-6`.

**Comparison:** Against baseline run `jgs4c6kl` (G=2) — see [RUNS.md](RUNS.md).

**Metrics:** Eval reward mean, `rewards/eval/check_answer`, KL (`actor/train/kl`), grad-norm, completion length, empty-completion rate, format-match rewards.

**Result:** Completed. CPU-fallback exact eval recovered `38/64 = 59.38%` for the restored checkpoint, with `31/64 = 48.44%` base-model exact accuracy on the same eval script. W&B scalar summary at step 3364 showed eval reward mean `1.908` and eval score mean `7.633`.

**Interpretation:** The `G=8, LR=3e-6` run is materially stronger than the recorded `G=8, LR=1e-5` run on the recovered exact GSM8K eval and on the final reward scalars, so the lower learning rate looks like the better setting in this branch.

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
