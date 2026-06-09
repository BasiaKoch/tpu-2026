# Experiments

Pre-register each experiment before launching a full TPU run.

## Experiment Summary

| Experiment | Main change | Hypothesis | Success metric | Diagnostics | Status |
|---|---|---|---|---|---|
| Baseline GRPO | Default `scripts/config.py` and rewards. | Baseline should reproduce the expected reward/KL behaviour and improve over the base model. | GSM8K eval accuracy. | Reward curve, KL curve, output format accuracy. | completed; LoRA eval below base on 64-example greedy eval |
| Variant A: reward/length | Add a mild length penalty or adjust shaping reward. | Reducing verbosity or reward hacking may improve correctness without excessive KL drift. | GSM8K eval accuracy vs. baseline. | Response length, malformed-output rate, correctness reward. | planned |
| Variant B: KL control | Conservative change to beta or epsilon. | KL budget should trade off stability against policy movement. | Accuracy read alongside KL. | KL vs. reward and KL vs. accuracy. | planned |

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

