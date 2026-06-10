"""Bootstrap a 95% confidence interval for eval accuracy.

Why this exists: I.3 artefact (a) asks for a measure of uncertainty on the
GSM8K accuracy. With a single eval seed we get a single accuracy number; the
bootstrap tells us how much that number would wobble if we had drawn a
different sample of test questions.

Input: the JSONL written by `evaluate.py --dump-per-question <path>` — one row
per question, each with a boolean field (default "correct").

Method (standard percentile bootstrap):
  1. Read the per-question 0/1 results.
  2. Resample the questions WITH REPLACEMENT, same count, many times.
  3. Each resample gives one accuracy; take the 2.5th and 97.5th percentiles
     of those accuracies as the 95% CI.

Run as:
    python analysis/bootstrap_eval_ci.py runs/<run>/per_question.jsonl
    python analysis/bootstrap_eval_ci.py path.jsonl --field partial --n-boot 10000 --out summary.json
"""
import argparse
import json

import numpy as np


def load_flags(path, field):
    """Read the JSONL file and return a list of 0/1 ints for the chosen boolean field."""
    flags = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            flags.append(1 if row[field] else 0)
    return flags


def bootstrap_ci(flags, n_boot, seed):
    """Return (observed accuracy, CI low, CI high, n) using a percentile bootstrap."""
    rng = np.random.default_rng(seed)   # fixed seed -> reproducible CI
    flags = np.array(flags)
    n = len(flags)
    observed = flags.mean()             # accuracy on the real sample

    boot_acc = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)   # n indices drawn with replacement
        boot_acc[b] = flags[idx].mean()

    lo, hi = np.percentile(boot_acc, [2.5, 97.5])
    return observed, lo, hi, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="per-question JSONL from evaluate.py --dump-per-question")
    ap.add_argument("--field", default="correct",
                    help="boolean field to score: correct (default), partial, or format")
    ap.add_argument("--n-boot", type=int, default=10000, help="number of bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0, help="fixed RNG seed for reproducibility")
    ap.add_argument("--out", default=None, help="optional path to save a JSON summary")
    args = ap.parse_args()

    flags = load_flags(args.jsonl, args.field)
    if not flags:
        raise SystemExit(f"No rows found in {args.jsonl}")

    acc, lo, hi, n = bootstrap_ci(flags, args.n_boot, args.seed)

    print(f"file       : {args.jsonl}")
    print(f"field      : {args.field}")
    print(f"n examples : {n}")
    print(f"accuracy   : {acc*100:.2f}%")
    print(f"95% CI     : [{lo*100:.2f}%, {hi*100:.2f}%]  "
          f"({args.n_boot} resamples, seed {args.seed})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "file": args.jsonl,
                "field": args.field,
                "n": n,
                "accuracy": acc,
                "ci_low": lo,
                "ci_high": hi,
                "n_boot": args.n_boot,
                "seed": args.seed,
            }, f, indent=2)
        print(f"wrote summary to {args.out}")


if __name__ == "__main__":
    main()
