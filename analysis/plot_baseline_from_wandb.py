"""Plot baseline I.1 reward and KL curves from a W&B history CSV export.

Reproducible companion to ``plot_baseline_curves.py`` (which reads TensorBoard
event files local to the machine that trained). This version consumes the CSV
produced by ``wandb_scripts/export_wandb_run.py`` so the baseline curves can be
regenerated from the shared W&B run (jgs4c6kl) on any machine.

    python analysis/plot_baseline_from_wandb.py \
        --csv wandb_scripts/data/baseline_jgs4c6kl_history.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_CSV = Path("wandb_scripts/data/baseline_jgs4c6kl_history.csv")
DEFAULT_OUT_DIR = Path("report_assets")

SERIES = {
    "baseline_mean_reward_curve": {
        "tag": "rewards/train/mean",
        "ylabel": r"Mean reward $\bar{r}$",
        "title": "Baseline GRPO training reward (run jgs4c6kl)",
    },
    "baseline_kl_curve": {
        "tag": "actor/train/kl",
        "ylabel": r"KL$(\pi_\theta \parallel \pi_{ref})$",
        "title": "Baseline KL divergence (run jgs4c6kl)",
    },
}


def load_series(csv_path: Path, tag: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if tag not in reader.fieldnames:
            raise ValueError(f"Tag {tag!r} not in CSV. Columns: {reader.fieldnames}")
        for row in reader:
            step_raw, val_raw = row.get("_step", ""), row.get(tag, "")
            if step_raw == "" or val_raw == "":
                continue
            try:
                rows.append((int(float(step_raw)), float(val_raw)))
            except ValueError:
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def write_csv(path: Path, rows: list[tuple[int, float]], value_name: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", value_name])
        writer.writerows(rows)


def write_plot(path_base: Path, rows: list[tuple[int, float]], ylabel: str, title: str) -> None:
    steps = [s for s, _ in rows]
    values = [v for _, v in rows]

    # light rolling mean to make the trend legible over noisy per-step values
    window = max(1, len(values) // 100)
    smooth = [
        sum(values[max(0, i - window): i + 1]) / len(values[max(0, i - window): i + 1])
        for i in range(len(values))
    ]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, values, linewidth=0.6, alpha=0.35, label="per step")
    plt.plot(steps, smooth, linewidth=1.8, label=f"rolling mean (w={window + 1})")
    plt.xlabel("GRPO step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), dpi=200)
    plt.savefig(path_base.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in SERIES.items():
        rows = load_series(args.csv, cfg["tag"])
        write_csv(args.out_dir / f"{name}.csv", rows, cfg["tag"])
        write_plot(args.out_dir / name, rows, cfg["ylabel"], cfg["title"])
        last = rows[-1] if rows else ("n/a", "n/a")
        print(f"wrote {name}: {len(rows)} points, final step={last[0]} value={last[1]}")


if __name__ == "__main__":
    main()
