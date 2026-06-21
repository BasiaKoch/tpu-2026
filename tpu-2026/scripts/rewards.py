"""Reward functions for GRPO on GSM8K.

Reward shaping rationale
------------------------
Pure correctness reward is very sparse: the model must produce a parseable
number AND get it right. Early in training it does neither, so the gradient
signal is near-zero. We add cheap shaping rewards that are non-zero almost
immediately:

  1. match_format_exactly        — full template match  (entry fee)
  2. match_format_approximately  — count of expected tags (dense, easy)
  3. check_answer                — graded numeric correctness (the prize)
  4. check_numbers               — fallback that just extracts a number

The total per-rollout reward is the sum across all four. GRPO then normalises
this within the group of G rollouts to compute the advantage. This is the
"group-relative" part of GRPO: no value network is learned; advantages come
from comparing siblings drawn from the same prompt.

Re-weighting strategy (see docs/rowan/reward-hacking-format-vs-accuracy.md)
--------------------------------------------------------------------------
Run 4i8lcitv reward-hacked format: the original design paid 5.5/rollout for
format (easy, dense) vs 4.5 for correctness (hard, sparse). Format saturated at
~96% by step ~1100, then absorbed all remaining optimisation pressure while
accuracy plateaued at ~51% of its maximum. The fixes below shift the balance so
correctness is the only axis with meaningful headroom left to optimise:

  1. Densify correctness.  check_answer used a cliff-shaped ladder
     (3.0 / 1.5 / 0.5 / 0.25 / -1.0). Replace it with a smooth
     CORRECT_MAX * exp(-CORRECT_K * relative_error) curve so even a near-miss
     has a gradient pointing *toward* the right answer — dense signal without
     making correctness binary.
  2. Shrink format to an entry fee.  match_format_exactly 3.0 -> 0.5 and
     match_format_approximately rescaled to +/-0.5 total, so the most a
     perfectly-formatted-but-wrong rollout can bank is ~1.0.
  3. Make correctness the prize.  The graded curve maxes at CORRECT_MAX (5.0) —
     ~5x the entire format budget — so the within-group advantage that drives
     GRPO is dominated by *correctness* differences between siblings rather than
     tiny *format* differences once format saturates.
  4. Keep format->answer gating.  check_answer / check_numbers still return 0
     unless the template parses, so reward cannot be accumulated on formatting
     alone.
  5. Fold check_numbers into the graded curve.  It overlaps check_answer; here it
     reuses the same curve but capped at a small CHECK_NUMBERS_MAX (0.5) as a
     looser-parse bootstrap, rather than a redundant 1.5 that re-inflates the
     already-saturated answer budget.
  6. (optional) Anneal format to zero.  Set FORMAT_ANNEAL_STEPS>0 to linearly
     decay the format coefficient 1.0 -> 0.0, removing the saturated reward
     entirely once it has done its bootstrap job. Off by default — the static
     rebalance above already does most of the work, and the step counter is
     approximate (see _format_coef).

Every magnitude is an env-overridable constant so a run can sweep them without a
code edit, matching the config.py convention (e.g. `CORRECT_MAX=4.0 python
train.py`).
"""
import math
import os
import re

from data import reasoning_start, reasoning_end, solution_start, solution_end


# ====== Re-weighting knobs (override per run via env) ======
# Correctness (the prize): graded reward = CORRECT_MAX * exp(-CORRECT_K * err),
# where err is the relative error |guess - true| / |true|. Larger CORRECT_K =
# sharper falloff (less credit for far-off answers); CORRECT_K=3 gives ~0.05*max
# at 100% relative error and ~0.74*max at 10% off.
CORRECT_MAX = float(os.environ.get("CORRECT_MAX", "5.0"))
CORRECT_K = float(os.environ.get("CORRECT_K", "3.0"))

# Format (the entry fee): kept small so it cannot dominate the advantage.
FORMAT_EXACT = float(os.environ.get("FORMAT_EXACT", "0.5"))   # full template parses
FORMAT_TAG = float(os.environ.get("FORMAT_TAG", "0.1"))       # per-tag, 5 tags -> +/-0.5

# check_numbers fallback: same graded curve, capped small to avoid double-paying
# correctness that check_answer already grades when the full template parses.
CHECK_NUMBERS_MAX = float(os.environ.get("CHECK_NUMBERS_MAX", "0.5"))

# Optional linear format annealing. 0 disables it (default). When >0, the format
# coefficient decays 1.0 -> 0.0 over this many reward-manager calls. NOTE: the
# counter increments once per call to check_numbers (the last reward fn), which
# tracks the training step closely but also advances on eval rollouts, so treat
# the schedule as approximate and set the value generously.
FORMAT_ANNEAL_STEPS = int(os.environ.get("FORMAT_ANNEAL_STEPS", "0"))


match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL,
)


# ====== Annealing bookkeeping ======
_call_count = 0


def _format_coef() -> float:
    """Current format multiplier in [0, 1] for the linear anneal schedule."""
    if FORMAT_ANNEAL_STEPS <= 0:
        return 1.0
    return max(0.0, 1.0 - _call_count / FORMAT_ANNEAL_STEPS)


def _graded_correctness(guess, true, max_reward) -> float:
    """Smooth partial credit in [0, max_reward] as a function of relative error.

    Exact string match short-circuits to the full reward (covers non-numeric or
    formatting-equal answers); otherwise a numeric relative error feeds the
    exp(-CORRECT_K * err) curve so near-misses get continuously more credit as
    they approach the true answer. Unparseable guesses earn 0 — GRPO's
    within-group normalisation supplies the contrast against correct siblings,
    so no explicit negative penalty is needed.
    """
    if guess is None:
        return 0.0
    if guess.strip() == true.strip():
        return max_reward
    try:
        err = abs(float(guess) - float(true)) / max(abs(float(true)), 1e-6)
    except Exception:
        return 0.0
    return max_reward * math.exp(-CORRECT_K * err)


def match_format_exactly(prompts, completions, **kwargs):
    """+FORMAT_EXACT if the whole template parses, 0 otherwise (entry fee)."""
    coef = _format_coef()
    return [
        0.0 if match_format.search(r) is None else FORMAT_EXACT * coef
        for r in completions
    ]


def match_format_approximately(prompts, completions, **kwargs):
    """Dense format bootstrap: +/-FORMAT_TAG per expected tag (max +/-0.5)."""
    coef = _format_coef()
    scores = []
    for response in completions:
        s = 0.0
        s += FORMAT_TAG if response.count(reasoning_start) == 1 else -FORMAT_TAG
        s += FORMAT_TAG if response.find(reasoning_start) == 0 else -FORMAT_TAG
        s += FORMAT_TAG if response.count(reasoning_end) == 1 else -FORMAT_TAG
        s += FORMAT_TAG if response.count(solution_start) == 1 else -FORMAT_TAG
        s += FORMAT_TAG if response.count(solution_end) == 1 else -FORMAT_TAG
        scores.append(s * coef)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    """Graded correctness of the bracketed answer (the prize, max CORRECT_MAX).

    Returns 0 when the template does not parse, so this reward is gated on format
    but — unlike format — never saturates while answers remain wrong.
    """
    extracted = [
        guess.group(1) if r is not None and (guess := match_format.search(r)) is not None else None
        for r in completions
    ]
    assert len(extracted) == len(answer)
    return [_graded_correctness(g, t, CORRECT_MAX) for g, t in zip(extracted, answer)]


def check_numbers(prompts, completions, answer, **kwargs):
    """Fallback bootstrap: graded credit (max CHECK_NUMBERS_MAX) on any number
    after <answer>, even when the full template does not parse."""
    global _call_count
    question = kwargs["question"]
    extracted = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in completions
    ]

    print("START ============================")
    print(f"Question:\t{question[0]}")
    print(f"Answer:\t{answer[0]}")
    print(f"Response:\t{completions[0]}")
    print(f"Extracted:\t{extracted[0]}")
    print("END ==============================")

    scores = [_graded_correctness(g, t, CHECK_NUMBERS_MAX) for g, t in zip(extracted, answer)]

    # Advance the anneal step once per reward-manager call (this fn is last in
    # REWARD_FNS, so the two format fns above already saw a consistent _call_count
    # for this step before it ticks forward).
    _call_count += 1
    return scores


REWARD_FNS = [match_format_exactly, match_format_approximately, check_answer, check_numbers]
