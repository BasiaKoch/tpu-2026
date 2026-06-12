"""Export baseline I.1 reward and KL curves from TensorBoard events."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_EVENT_FILE = Path(
    "/tmp/content/tmp/tensorboard/grpo/events.out.tfevents.1780942641.t1v-n-f769470f-w-0"
)
DEFAULT_OUT_DIR = Path("report_assets")

SERIES = {
    "baseline_mean_reward_curve": {
        "tag": "rewards/train/mean",
        "ylabel": "Mean reward",
        "title": "Baseline GRPO training reward",
    },
    "baseline_kl_curve": {
        "tag": "actor/train/kl",
        "ylabel": "KL(pi_theta || pi_ref)",
        "title": "Baseline KL divergence",
    },
}


def load_series(event_file: Path, tag: str) -> list[tuple[int, float]]:
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        raise ValueError(f"Scalar tag {tag!r} not found. Available tags: {sorted(tags)}")
    return [(point.step, point.value) for point in accumulator.Scalars(tag)]


def write_csv(path: Path, rows: list[tuple[int, float]], value_name: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", value_name])
        writer.writerows(rows)


def write_plot(path_base: Path, rows: list[tuple[int, float]], ylabel: str, title: str) -> None:
    steps = [step for step, _ in rows]
    values = [value for _, value in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, values, linewidth=1.6)
    plt.xlabel("GRPO step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), dpi=200)
    plt.savefig(path_base.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-file", type=Path, default=DEFAULT_EVENT_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.event_file.exists():
        raise FileNotFoundError(args.event_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in SERIES.items():
        rows = load_series(args.event_file, cfg["tag"])
        write_csv(args.out_dir / f"{name}.csv", rows, cfg["tag"])
        write_plot(args.out_dir / name, rows, cfg["ylabel"], cfg["title"])
        print(f"wrote {name}: {len(rows)} points")


if __name__ == "__main__":
    main()
