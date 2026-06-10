"""Overlay training curves for the baseline and variants on shared axes.

Why this exists: I.3 artefact (b) wants mean-reward and KL curves for the
baseline AND the variants on ONE set of axes; artefact (c) wants one diagnostic
of a failure mode. The completion-length panel below is that diagnostic — it
shows the length blow-up / collapse described in docs/rowan/baseline-analysis.md.

Input: one CSV per run. Each CSV must have a `step` column and one or more
metric columns. The filename stem is used as the run label unless you pass
--labels. This matches the `step,<tag>` CSVs that plot_baseline_curves.py
already writes, and the multi-column CSVs you get from a W&B history export.

For each of the four metrics below we look for a known column name in every
run's CSV and overlay the runs that have it. Missing metrics/runs are skipped
with a printed message — the script never invents data.

Run as:
    python analysis/plot_training_curves.py baseline.csv length_penalty.csv
    python analysis/plot_training_curves.py a.csv b.csv --labels baseline length-penalty
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Each entry: (output filename, y-axis label, accepted column names in priority order).
# The first name found in a run's CSV is used. If your export names a column
# differently, add it here — the script prints each file's columns so you can see
# what is actually available.
METRICS = [
    ("mean_reward", "Mean train reward",
     ["rewards/train/mean", "mean_reward", "reward"]),
    ("kl", "KL(pi_theta || pi_ref)",
     ["actor/train/kl", "kl"]),
    ("eval_reward", "Eval mean reward",
     ["rewards/eval/mean", "eval_reward"]),
    ("completion_length", "Mean completion length",
     ["rewards/eval/completion_length", "completion_length", "response_length_mean"]),
]


def read_csv(path):
    """Return (columns, data) where data maps each column to a list of floats (None if not a number)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        data = {c: [] for c in columns}
        for row in reader:
            for c in columns:
                try:
                    data[c].append(float(row[c]))
                except (ValueError, TypeError):
                    data[c].append(None)
    return columns, data


def pick_column(columns, candidates):
    """Return the first candidate column name present in `columns`, else None."""
    for name in candidates:
        if name in columns:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="one CSV per run (each needs a 'step' column)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="run labels, one per CSV (default: the filename stem)")
    ap.add_argument("--step-col", default="step", help="name of the step column")
    ap.add_argument("--out-dir", default="analysis/figures", help="where to save the figures")
    args = ap.parse_args()

    labels = args.labels if args.labels else [Path(p).stem for p in args.csvs]
    if len(labels) != len(args.csvs):
        raise SystemExit(f"got {len(args.csvs)} CSVs but {len(labels)} labels")

    # Load every run once and print its columns, so a missing metric is obvious.
    runs = []
    for path, label in zip(args.csvs, labels):
        columns, data = read_csv(path)
        if args.step_col not in columns:
            raise SystemExit(f"{path}: no '{args.step_col}' column. Found: {columns}")
        print(f"{label}: columns = {columns}")
        runs.append((label, columns, data))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for out_name, ylabel, candidates in METRICS:
        plt.figure(figsize=(7, 4))
        plotted = 0
        for label, columns, data in runs:
            col = pick_column(columns, candidates)
            if col is None:
                continue
            # keep only rows where both the step and the value are real numbers
            pairs = [(s, v) for s, v in zip(data[args.step_col], data[col])
                     if s is not None and v is not None]
            if not pairs:
                continue
            xs = [s for s, _ in pairs]
            ys = [v for _, v in pairs]
            plt.plot(xs, ys, linewidth=1.5, label=f"{label} ({col})")
            plotted += 1

        if plotted == 0:
            plt.close()
            print(f"skipping {out_name}: none of {candidates} found in any input CSV")
            continue

        plt.xlabel("GRPO step")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs step")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{out_name}.png", dpi=200)
        plt.savefig(out_dir / f"{out_name}.pdf")
        plt.close()
        print(f"wrote {out_name} ({plotted} run(s)) to {out_dir}")


if __name__ == "__main__":
    main()
