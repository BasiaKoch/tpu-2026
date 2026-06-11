# Experiments

Pre-register each experiment before launching a full TPU run.

## Experiment Summary

| Experiment | Main change | Hypothesis | Success metric | Diagnostics | Status |
|---|---|---|---|---|---|
| Baseline GRPO | Default `scripts/config.py` and rewards. | Baseline should reproduce the expected reward/KL behaviour and improve over the base model. | GSM8K eval accuracy. | Reward curve, KL curve, output format accuracy. | completed; LoRA eval below base on 64-example greedy eval |
| Variant A: reward/length | Add a mild length penalty or adjust shaping reward. | Reducing verbosity or reward hacking may improve correctness without excessive KL drift. | GSM8K eval accuracy vs. baseline. | Response length, malformed-output rate, correctness reward. | planned |
| Variant B: KL control | Conservative change to beta or epsilon. | KL budget should trade off stability against policy movement. | Accuracy read alongside KL. | KL vs. reward and KL vs. accuracy. | planned |
| Variant A2: empty-completion penalty | Add `empty_penalty` (-2.0 for completions < 20 chars) to `REWARD_FNS`; K=2, else default. | The collapse attractor is the empty completion (reward-flat + KL-free); penalising near-empty outputs makes the attractor's entrance repulsive while groups are still mixed, delaying or preventing the terminal collapse. | Collapse onset step + GSM8K eval accuracy vs. baseline `jgs4c6kl`. | Completion length vs. step, empty-completion rate, KL, eval reward; collapse step vs. baseline (~2700) and vs. length-penalty run (~2750). | registered (`empty-penalty`) |

## Logged Experiments

### `empty-penalty`

**Question:** Mechanism-targeted follow-up to the length-penalty and beta runs: if the terminal failure mode is the empty-output absorbing state, does directly penalising near-empty completions prevent entry into it?

**Main change:** One new reward function in [scripts/rewards.py](../scripts/rewards.py): `empty_penalty` = -2.0 for completions shorter than 20 characters (stripped), 0 otherwise; appended to `REWARD_FNS`. Nothing else changes (K=2, beta=0.08, LR 3e-6, bs=1, default config).

**Motivation (from this week's runs):** Baseline (`jgs4c6kl`), beta=0.32 (`oet2tfjd`), and the K=2 length-penalty run (`jcp0b5cy`) all terminate in the same state: empty completions, reward frozen, sigma_r = 0, KL = 0, gradient = 0. The state is reward-flat under the existing terms — an empty string and a tagless non-empty string both score -2.5 — and KL-free (no tokens), so no loss term resists it. The length penalty capped the length blowup (max ~249 vs ~471) but could not see the empty attractor (length 0 ⇒ penalty 0), and the run collapsed at ~step 2750 anyway. `empty_penalty` makes a near-empty completion strictly worse (-4.5) than any non-empty sibling (≥ -2.5), so in *mixed* groups empty completions receive negative advantage and the policy is pushed away from the attractor's entrance.

**Pre-registered caveat:** once a group is fully empty, all K rewards are again equal (sigma_r = 0) and *no* reward function can generate a gradient — the penalty acts only during the descent. If the run still collapses at a similar step, that is evidence the descent at K=2 is too abrupt for reward shaping to catch, strengthening the conclusion that only the advantage-signal quality (group size K) addresses the K=2 collapse.

**Config:** Branch `empty-penalty-bk` (off `origin/baseline-fls`); K=2 baseline config throughout; data shuffle seed 42; `MAX_STEPS=3364`. Control: baseline `jgs4c6kl`; secondary read against the length-penalty run `jcp0b5cy` (same VM).

**Metrics:** Collapse onset step (primary), empty-completion rate, completion length vs. step, eval reward, KL, GSM8K accuracy across checkpoints.

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

