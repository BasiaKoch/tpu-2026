# Experiments

Pre-register each experiment before launching a full TPU run.

## Experiment Summary

| Experiment | Main change | Hypothesis | Success metric | Diagnostics | Status |
|---|---|---|---|---|---|
| Baseline GRPO | Default `scripts/config.py` and rewards. | Baseline should reproduce the expected reward/KL behaviour and improve over the base model. | GSM8K eval accuracy. | Reward curve, KL curve, output format accuracy. | completed; LoRA eval below base on 64-example greedy eval |
| Variant A: reward/length | Add a mild length penalty or adjust shaping reward. | Reducing verbosity or reward hacking may improve correctness without excessive KL drift. | GSM8K eval accuracy vs. baseline. | Response length, malformed-output rate, correctness reward. | planned |
| Variant B: KL control | Conservative change to beta or epsilon. | KL budget should trade off stability against policy movement. | Accuracy read alongside KL. | KL vs. reward and KL vs. accuracy. | planned |

## Variant: reward-length-bk

Owner: Barbara Koch  
Branch: `reward-length-bk`  
Base branch: `origin/baseline-fls`  
Primary file to change later: `scripts/rewards.py`

### Hypothesis
The baseline collapse may be partly caused by response-length blowup and reward-shaping instability. A mild excess-length penalty may reduce verbosity and reward hacking while keeping the rest of the GRPO setup unchanged.

### Planned change
Add one new reward function, `length_penalty`, to `scripts/rewards.py` and append it to `REWARD_FNS`.

The planned penalty should:
- use the same reward-function interface as the existing functions;
- measure length simply using `len(completion)` to avoid tokenizer dependencies;
- apply no penalty below a target length;
- apply only a capped, mild negative penalty above the target length;
- keep all existing reward functions unchanged.

### Controlled comparison
Keep unchanged:
- model;
- data;
- LoRA rank;
- group size `K`;
- beta / KL coefficient;
- epsilon / clipping range;
- learning rate;
- generation length;
- number of training steps;
- evaluation protocol.

### Diagnostics
Compare against the baseline using:
- mean reward vs GRPO step;
- KL vs GRPO step;
- response length vs step;
- GSM8K accuracy across checkpoints, not only the final checkpoint.

### Risk
If the penalty is too strong, it may suppress reasoning or encourage overly short/empty answers. This is important because the existing baseline reportedly showed both length blowup and later collapse.

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

