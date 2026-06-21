# W&B Training Metrics Reference

Every metric you see on the W&B dashboard during a GRPO run is produced by **tunix** (the
training library), not by anything in this repo. `scripts/train.py` just calls
`wandb.init(...)` and then hands control to `GRPOLearner.train()`; tunix logs everything from
there. This doc explains what each metric means and, in particular, disentangles the
similarly-named ones (e.g. `rewards/train/score/mean` vs `rewards/train/mean`).

## How a metric name is built

Internally tunix logs every scalar with three pieces:

```
{prefix}/{mode}/{metric_name}
```

- **prefix** — the family the metric belongs to: `rewards`, `actor`, `completions`, …
- **mode** — `train` or `eval`. `eval` metrics are logged on the held-out validation set
  every `EVAL_EVERY_N_STEPS` steps; `train` metrics come from the rollouts the model is
  actually being trained on.
- **metric_name** — the leaf name, which may itself contain slashes (e.g. `score/mean`).

The full string is what lands in W&B verbatim (leading slashes stripped). The `step` axis is
the global training step.

Two code paths assemble these names, which is the source of the confusing near-duplicates:

1. `metrics_logger.log(prefix, name, value, mode, step)` is called directly with an explicit
   prefix. → `peft_trainer.py` (the `actor/*` core metrics).
2. `rl_cluster.buffer_metrics({"rewards/score/mean": ...}, mode)` passes a key that already
   contains slashes; `_log_metrics` then **splits off the first segment as the prefix** and
   inserts the mode after it. So the buffered key `rewards/score/mean` with `mode=train`
   becomes `rewards/train/score/mean`. (`rl_cluster.py:735`)

That second rule is why `rewards/score/mean` (buffered key) and `rewards/mean` (buffered key)
end up as `rewards/train/score/mean` and `rewards/train/mean` — same prefix, different leaf,
and they mean genuinely different things (see below).

---

## `rewards/*` — the reward signal

This is the most important family and also the most confusing, because there are **two
different reductions of two different quantities** living under the same prefix.

### `rewards/train/score/{mean,max,min}`

Source: `grpo_learner.py:324`. Logged from the array of **total per-rollout rewards** — i.e.
for each generated completion, the sum across all four reward functions
(`match_format_exactly + match_format_approximately + check_answer + check_numbers`). This is
the scalar that actually drives the advantage computation.

- `score/mean` — mean total reward across every rollout in the batch
  (batch = `prompts × NUM_GENERATIONS`).
- `score/max` / `score/min` — best / worst single rollout in the batch.

**This is your headline "is the model getting better" curve.** It should climb over training.

### `rewards/train/{sum,mean,min,max}`

Source: `reward_manager.py:_calculate_scalar_reward_log_metrics`, called with
`prefix="rewards"`. These come from the **same per-rollout total reward**, but the reductions
are computed differently:

- The raw `rewards` array here is shaped `(num_prompts, num_reward_fns)`. `sum`/`mean`/`min`/
  `max` are taken **along the reward-function axis first** (axis=1), giving one number per
  prompt, and then averaged across prompts.
- `rewards/sum` ≈ the mean over prompts of the *summed* reward (sum over the four functions).
  In practice `rewards/sum` tracks very closely with `rewards/score/mean` because both are
  "total reward per rollout, averaged" — they're computed in different places (reward manager
  vs. GRPO learner) over slightly different groupings, but they measure the same underlying
  thing.
- `rewards/mean` is the *per-reward-function* mean (divides the sum by the number of functions,
  4), so it sits at roughly `rewards/sum ÷ 4`. **Do not read `rewards/mean` as the overall
  reward** — it's the average contribution of a single reward function, which is much smaller.
- `rewards/min` / `rewards/max` are the min/max reward-function contribution, then averaged
  over prompts — not the same as `score/min` / `score/max`, which are the extreme *rollouts*.

> **TL;DR for the duplicates:** use `rewards/train/score/mean` as the primary reward curve.
> `rewards/train/sum` is an equivalent "total reward" view. `rewards/train/mean` is a *per-
> function* average (≈ sum/4) and is easy to misread as low performance.

### `rewards/train/{match_format_exactly, match_format_approximately, check_answer, check_numbers}`

Source: `reward_manager.py:215`. One curve per reward function in `REWARD_FNS`, named after the
Python function. Each is the **batch-mean of that single function's contribution**. These let
you see *which* shaping term is firing:

- `match_format_exactly` — 0 or 3.0; rises once the model learns the full XML template.
- `match_format_approximately` — up to +2.5; dense tag-counting signal, moves early.
- `check_answer` — partial-credit correctness (−1.0 … +3.0); the one you ultimately care about.
- `check_numbers` — fallback numeric-match reward (0 or 1.5).

`rewards/score/mean` ≈ the sum of these four curves (modulo batching/aggregation). If
`score/mean` is climbing but `check_answer` is flat, the model is winning reward by formatting,
not by solving — see `baseline-eval-format-gap.md` for exactly this failure mode.

### `rewards/eval/*`

The identical set of reward metrics, computed on the validation set. Compare `rewards/eval/*`
against `rewards/train/*` to spot overfitting to the training rollouts.

---

## `actor/*` — the policy-model / optimizer metrics

Source: `peft_trainer.py` (`metrics_prefix = "actor"`, set at `rl_cluster.py:565`). These
describe the gradient update applied to the LoRA policy ("actor") model each step.

- `actor/train/loss` — the GRPO objective being minimised (`grpo_loss_fn`, `algo_core.py:342`).
  This is the clipped policy-gradient loss plus `BETA × KL` penalty. It is **not** a quality
  metric — GRPO loss hovers near zero and is noisy; do not read "loss going down" as "model
  improving". Watch `rewards/score/mean` for quality.
- `actor/train/perplexity` — `exp(loss)`. Same caveat; mostly a transform of loss.
- `actor/train/learning_rate` — current LR from the warmup-cosine schedule (`build_optimizer`).
  Useful for confirming warmup/decay is behaving.
- `actor/train/grad_norm` — global L2 norm of the gradients **before** clipping
  (`MAX_GRAD_NORM`). Spikes here flag instability; if it sits at the clip threshold the update
  is being capped.
- `actor/train/kl` — mean KL divergence between the current policy and the frozen **reference**
  model over completion tokens (`algo_core.py:530`). This is the regulariser keeping the policy
  from drifting too far from the base model. Only computed/non-zero when `BETA != 0`. Rising KL
  = policy moving away from the base model.
- `actor/train/pg_clipfrac` — fraction of tokens where the PPO/GRPO importance ratio hit the
  `EPSILON` clip bound (`algo_core.py:440`). High values mean the new policy is far from the
  rollout policy and updates are being clipped — a sign the LR or batch dynamics are too
  aggressive.

### `actor/eval/*`

`actor/eval/loss`, `actor/eval/perplexity`, `actor/eval/kl`, `actor/eval/pg_clipfrac` — same
quantities on the validation set. Note **`grad_norm` and `learning_rate` are train-only**:
eval does no backward pass and applies no optimizer step, so those have no eval counterpart.

> **`actor/train/kl` vs the KL inside `actor/train/loss`:** the loss term uses `BETA × kl_loss`
> (a per-token aggregation), while `actor/train/kl` is the plain mean KL logged for monitoring.
> They move together but are not numerically identical, and if `BETA == 0` the loss contains no
> KL term at all while `actor/train/kl` stays 0.0.

---

## `completions/*` — output length

Source: `grpo_learner.py:335`. Length (in tokens, via the completion mask) of the generated
responses:

- `completions/train/mean_length`, `completions/train/max_length`,
  `completions/train/min_length`.

Watch these for two failure modes: collapse toward `min_length` (the model learns to emit a
terminator early to game a reward) or saturation at `max_length` =
`TOTAL_GENERATION_STEPS` (responses being truncated, which can starve `check_answer`).

---

## What is *not* on W&B

A few things are easy to expect but won't appear:

- **Eval accuracy / format accuracy.** `evaluate.py` does **not** log to W&B. The
  `eval/base_acc`, `eval/base_format`, … keys referenced by the run-checklist tooling are added
  manually via `wandb.run.summary.update()`, not emitted during training. (`run_checklist_data.py:410`)
- **Most of the GRPO loss diagnostics.** `grpo_loss_fn` computes a rich `aux` dict
  (`is_ratio/*`, `advantage/*`, `ppo_kl`, `entropy`, `pg_loss/*`, …, `algo_core.py:493`), but
  tunix only forwards two of them to the logger — `kl` and `pg_clipfrac`, via
  `with_rl_metrics_to_log({...})` at `grpo_learner.py:197`. The rest are computed and discarded
  unless that registration list is extended.

---

## Quick reference: which curve do I actually look at?

| Question | Metric |
|---|---|
| Is the model improving overall? | `rewards/train/score/mean` |
| Is it solving problems vs. just formatting? | `rewards/train/check_answer` vs `rewards/train/match_format_*` |
| Overfitting to train rollouts? | `rewards/eval/score/mean` vs `rewards/train/score/mean` |
| Training stable? | `actor/train/grad_norm`, `actor/train/pg_clipfrac` |
| Policy drifting from base model? | `actor/train/kl` |
| Responses collapsing / truncating? | `completions/train/{min,max,mean}_length` |

### Common pitfalls

- `rewards/train/mean` is **per-reward-function** (≈ total ÷ 4), not the overall reward. Use
  `score/mean` or `sum`.
- `actor/train/loss` going down does **not** mean the model is getting better. GRPO loss is
  near-zero and noisy by design; reward is the quality signal.
