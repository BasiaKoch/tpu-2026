# Reward Hacking: Format Over-Optimization vs. Answer Accuracy

Analysis of training run `4i8lcitv` (`wandb_scripts/data/wandb_4i8lcitv_history.csv`, 3,364
steps). Question: is the model reward hacking — driving up the total `score` by optimizing
output **format** without improving **answer accuracy**?

## Verdict: Yes — confirmed, with one nuance

The model over-optimizes format. Accuracy improves for the first ~1/3 of training, then
**plateaus for the remaining ~2,250 steps** while format keeps creeping toward saturation. The
same pattern appears on the held-out eval set, so it is a genuine optimization plateau, not
train-set memorization.

### Evidence 1 — format is saturated, accuracy is stuck at ~half

Reward captured as % of each component's maximum (final 250 train steps):

| Component | Mean | Max possible | % of max captured |
|---|---|---|---|
| `match_format_exactly` | 2.89 | 3.0 | **96%** |
| `match_format_approximately` | 2.40 | 2.5 | **96%** |
| `check_answer` | 1.53 | 3.0 | **51%** |
| `check_numbers` | 0.91 | 1.5 | **61%** |

Format is essentially maxed out (96%) with only 4% headroom left. `check_answer` — the real
accuracy proxy — sits at 51% of its maximum, leaving ~half the achievable correctness reward on
the table, untouched.

### Evidence 2 — the plateau (the smoking gun)

`check_answer` is **not** flat the whole run. It climbs early, then freezes:

| | step 0–250 | step ~1100 | step ~3300 |
|---|---|---|---|
| `check_answer` (train) | 0.53 | 1.51 | 1.53 |
| gain over prior window | — | **+0.98** | **+0.02** |

For the back two-thirds of training, accuracy gain is **+0.02** (noise), while format continues
inching up. In that window essentially **100% of the score increase is format polishing**.

Over the whole run, decomposing the total score gain (first 250 vs last 250 steps):

- **Format gain: 68%** (`match_format_exactly` +1.18, `match_format_approximately` +1.38)
- **Answer gain: 32%** (`check_answer` +1.00, `check_numbers` +0.21)

### Evidence 3 — eval confirms it's not memorization

Held-out eval shows the identical signature: `check_answer` mid 1.48 → late 1.53 (flat) while
`match_format_exactly` keeps rising (2.29 → 2.85). The plateau is a real capability/optimization
ceiling, not overfitting to the training rollouts.

## Root cause: the reward design rewards the easy axis too heavily

The four reward functions (`scripts/rewards.py`) split into two axes:

| Axis | Components | Max reward | Difficulty |
|---|---|---|---|
| **Format** | `match_format_exactly` (3.0) + `match_format_approximately` (2.5) | **5.5** (55% of total) | easy, dense |
| **Answer** | `check_answer` (3.0) + `check_numbers` (1.5) | **4.5** (45% of total) | hard, sparse |

Two specific flaws:

1. **Format is worth too much and is easy/dense.** The policy banks 96% of it first, and the
   gradient there dries up — but it keeps paying out 5.5/rollout for the rest of the run,
   providing no pressure toward accuracy.
2. **`match_format_exactly` (3.0) pays exactly as much as a perfectly correct answer (3.0).**
   Getting the XML tags right (trivial) is rewarded as much as solving the math (hard). Per unit
   of effort, format is hugely more attractive.

Once format saturates (~step 1100), the within-group advantage that drives GRPO is dominated by
tiny *format* differences between siblings rather than *correctness* differences — so there is
little signal pushing accuracy past ~51%.

## Proposed reward changes

Constraint: a perfectly correct answer is a **sparse** reward, so we can't just make correctness
binary or early training loses its bootstrap signal. The fix has two halves — **densify the
correctness gradient** *and* **stop paying for format once it's learned.**

### 1. Densify `check_answer` to fight sparsity (highest priority)

It currently has a cliff-shaped ladder (exact 3.0 → close 1.5 → ratio bands 0.5/0.25 → −1.0).
Replace the bands with a smooth function of relative error, so there is always a gradient
pointing *toward* the right answer even on a near-miss:

```python
err = abs(float(guess) - float(true)) / max(abs(float(true)), 1e-6)
score = CORRECT_MAX * math.exp(-k * err)   # e.g. CORRECT_MAX=5.0, k≈3
```

This gives continuously increasing partial credit as answers get closer — dense signal without
making correctness binary.

### 2. Rebalance magnitudes so correctness dominates

Shrink format to a small "entry fee" and make correctness the prize:

- `match_format_exactly`: 3.0 → **0.5**
- `match_format_approximately`: keep dense but cap total ~**0.5**
- `check_answer` exact: 3.0 → **~5.0** (via the curve in #1)

### 3. Anneal the format reward toward zero over training

Directly targets the observed plateau. Multiply the format rewards by a coefficient that decays
(e.g. linear 1.0 → 0.0 between step 0 and ~1000). Early on, dense format reward bootstraps;
afterward the *only* way to raise reward is to answer correctly, forcing the gradient onto
accuracy instead of letting the model bank format forever.

### 4. Strengthen format→answer gating (multiplicative, not additive)

`check_answer` already returns 0 when format doesn't parse, but the big format bonus is still
paid additively regardless of correctness. Restructure as
`total = format_bonus + (format_ok) * correctness` so the model cannot accumulate reward purely
on formatting.

### 5. Fold `check_numbers` into the graded correctness reward

It overlaps `check_answer` (both reward numeric correctness) and redundantly inflates the answer
budget that is already saturated at ~60%. Merge into the single graded curve from #1 for a
cleaner gradient.

### Highest-leverage combination

**#1 + #3.** Densifying correctness gives a usable gradient even on near-misses; annealing format
removes the saturated reward that is currently absorbing all the optimization pressure in the
back two-thirds of the run.

## Caveat to test

Part of the plateau at ~51% may be a genuine **capability ceiling** for Gemma-3-1B on GSM8K, not
just reward misallocation. If after rebalancing `check_answer` still plateaus at a similar level,
that points to model capacity rather than reward design. Worth ruling in/out with a short run
before investing further in reward tuning.

## Reproduction

```bash
source ~/venvs/tunix/bin/activate
# analysis performed directly on:
#   wandb_scripts/data/wandb_4i8lcitv_history.csv
# key columns: rewards/train/{score/mean, match_format_exactly,
#   match_format_approximately, check_answer, check_numbers} and the rewards/eval/* mirror.
```

See `wandb-training-metrics.md` for what each metric means and
`baseline-eval-format-gap.md` for the related accuracy-vs-format regex distinction.
