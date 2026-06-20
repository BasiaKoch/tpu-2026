"""Diagnostic metric functions for GRPO training.

These are passed to ``GRPOLearner(metric_fns=...)``. Tunix calls each metric fn
inside ``_generate_and_compute_advantage`` for BOTH train and eval rollouts, with
the signature ``(prompts, completions, rewards, advantages, **kwargs)`` where
``kwargs`` carries the per-rollout dataset fields (here: ``question``, ``answer``).
Each fn returns ``{metric_name: (value, agg_fn)}``; tunix buffers the values and
aggregates them with ``agg_fn`` once per step, logging to W&B as
``{prefix}/{train|eval}/{name}`` (the prefix is the part before the first "/").

Because the same fns run in eval mode, every metric below is automatically logged
under both ``diagnostics/train/...`` and ``diagnostics/eval/...``; the eval copy is
the per-step eval diagnostic we care about (eval fires every EVAL_EVERY_N_STEPS).

Why these can be computed here cheaply:
  * accuracy            — completions + gold ``answer`` are both in scope; we reuse
    the same numeric-match parse as evaluate.py (match_numbers).
  * degenerate_group_frac — ``rewards`` is the flat [B] vector of summed rewards,
    laid out as contiguous groups of NUM_GENERATIONS (GRPO repeats each prompt G
    times in both modes), so we can reshape and test within-group variance.

Entropy is NOT here: it is already computed inside the GRPO policy-loss as
``aux["entropy"]`` and is wired up in train.py via ``with_rl_metrics_to_log``,
which is cheaper (free from the loss) and avoids re-running the model.
"""
import numpy as np

from config import NUM_GENERATIONS
from rewards import match_numbers


def _extract_number(response: str):
    """Parse the first number after <answer>, mirroring evaluate.py."""
    guess = match_numbers.search(response)
    return guess.group(1) if guess is not None else None


def accuracy_metric(prompts, completions, rewards, advantages, **kwargs):
    """Fraction of completions whose extracted number exactly equals the gold answer.

    This counts every rollout (with G samples per prompt at the rollout
    temperature), so it is a *sampled* accuracy and will read lower than the
    final greedy single-pass eval in evaluate.py — but it tracks the same trend
    per step and, critically, gives us per-step EVAL accuracy alongside
    eval_reward_mean, which we previously did not log.
    """
    answer = kwargs["answer"]
    correct = 0
    total = 0
    for response, ans in zip(completions, answer):
        total += 1
        ext = _extract_number(response)
        if ext is None or ans is None:
            continue
        try:
            if float(ext.strip()) == float(str(ans).strip()):
                correct += 1
        except (ValueError, TypeError):
            pass
    acc = correct / total if total else 0.0
    return {"diagnostics/accuracy": (acc, np.mean)}


def degenerate_group_metric(prompts, completions, rewards, advantages, **kwargs):
    """Fraction of G-sized rollout groups with zero within-group reward variance.

    When all G rewards in a group are equal, the GRPO advantage (reward minus
    group mean, divided by group std) is ill-defined / zero — the group
    contributes no learning signal. A rising degenerate fraction is the direct
    fingerprint of the within-group-variance collapse that kills the K=2 run, so
    this is the diagnostic we most want to track.
    """
    r = np.asarray(rewards, dtype=np.float32).ravel()
    g = NUM_GENERATIONS
    if g <= 1 or r.size == 0 or r.size % g != 0:
        # Can't form complete groups; skip rather than log a misleading value.
        return {}
    groups = r.reshape(-1, g)
    # ptp == 0  <=>  all rewards in the group are identical (exact, since these
    # are deterministic float sums of the same reward fns). Tiny tolerance guards
    # against float round-off in the summation order.
    degenerate = np.ptp(groups, axis=1) <= 1e-8
    return {"diagnostics/degenerate_group_frac": (float(np.mean(degenerate)), np.mean)}


METRIC_FNS = [accuracy_metric, degenerate_group_metric]
