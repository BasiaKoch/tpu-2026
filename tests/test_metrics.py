#!/usr/bin/env python3
"""Synthetic unit tests for the diagnostic metric fns in scripts/metrics.py.

Why this exists (and why it matters): on the K=8 run the degenerate-group
fraction sits at ~0 throughout, and the eventual K=2 contrast hinges on it
*rising*. But "always 0" looks identical whether the detector is correct or
silently hard-wired to 0 by a bug. These tests pin the detector's behaviour with
synthetic rewards so a 0 reading can be trusted: an all-equal group MUST score
1.0 (degenerate), a fully-varied group 0.0, and a mix the right fraction.

Pure numpy — no TPU/model needed. Run:  python tests/test_metrics.py
(imports scripts/metrics.py, which imports config/rewards; the tunix venv has
numpy + jax, which is all that's required.)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from metrics import accuracy_metric, degenerate_group_metric  # noqa: E402
from config import NUM_GENERATIONS  # noqa: E402

G = NUM_GENERATIONS
KEY_DEG = "diagnostics/degenerate_group_frac"
KEY_ACC = "diagnostics/accuracy"


def _deg(rewards):
    """Return the degenerate-group fraction value for a flat rewards array."""
    out = degenerate_group_metric(None, [""] * len(rewards), np.asarray(rewards), None)
    return out.get(KEY_DEG, (None,))[0]


def test_all_equal_group_is_degenerate():
    # One group of G identical rewards -> the detector MUST fire (1.0).
    assert _deg([3.0] * G) == 1.0, "all-equal group should be 100% degenerate"


def test_varied_group_is_not_degenerate():
    # One group of G distinct rewards -> not degenerate (0.0).
    assert _deg(list(range(G))) == 0.0, "all-distinct group should be 0% degenerate"


def test_mixed_groups_fraction():
    # Two groups: one all-equal (degenerate), one varied -> 0.5.
    rewards = [2.0] * G + list(range(G))
    assert _deg(rewards) == 0.5, "one degenerate of two groups should be 0.5"


def test_float_tolerance():
    # Rewards equal up to tiny float round-off still count as degenerate.
    rewards = [1.0, 1.0 + 1e-12] + [1.0] * (G - 2)
    assert _deg(rewards) == 1.0, "within-tolerance group should be degenerate"


def test_incomplete_groups_skipped():
    # Size not a multiple of G -> metric is omitted rather than misreported.
    out = degenerate_group_metric(None, [""] * (G + 1), np.ones(G + 1), None)
    assert KEY_DEG not in out, "incomplete group batch should be skipped"


def test_accuracy_counts_numeric_matches():
    comps = [
        "<reasoning>x</reasoning><answer>42</answer>",   # correct
        "<reasoning>y</reasoning><answer>7</answer>",    # wrong
        "no answer tag here",                            # unparseable -> wrong
    ]
    answers = ["42", "13", "5"]
    acc = accuracy_metric(None, comps, np.zeros(3), None, answer=answers)[KEY_ACC][0]
    assert abs(acc - 1.0 / 3.0) < 1e-9, f"expected 1/3, got {acc}"


def test_accuracy_handles_float_equality():
    # "42.0" should match gold "42".
    comps = ["<reasoning>x</reasoning><answer>42.0</answer>"]
    acc = accuracy_metric(None, comps, np.zeros(1), None, answer=["42"])[KEY_ACC][0]
    assert acc == 1.0, "42.0 should match 42 numerically"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} metric tests passed (G={G}).")


if __name__ == "__main__":
    main()
