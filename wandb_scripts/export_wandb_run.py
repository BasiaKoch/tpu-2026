#!/usr/bin/env python3
"""Export a Weights & Biases training run for offline analysis.

This utility downloads the full scalar history for a W&B run as CSV, plus
the run summary, config, and basic metadata as JSON files. These files are the
easiest way to share GRPO training metrics with another tool or person without
granting access to the W&B project. By default, exports are written to the
`data` directory next to this script.

Configuration is intentionally flexible:

    python export_wandb_run.py
    python export_wandb_run.py --run-id bnh9ttlt
    python export_wandb_run.py --entity my-team --project tunix --run-id bnh9ttlt
    python export_wandb_run.py --run-path my-team/tunix/runs/bnh9ttlt

Before running, authenticate with `wandb login`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed in the active Python environment.")
    print("Activate the TPU environment, for example: source ~/venvs/tunix/bin/activate")
    sys.exit(1)


DEFAULT_ENTITY = "felsomoye-university-of-cambridge"
DEFAULT_PROJECT = "tunix"
DEFAULT_RUN_ID = "jgs4c6kl"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export W&B run history, summary, config, and metadata."
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=f"W&B run id, for example bnh9ttlt. Defaults to {DEFAULT_RUN_ID!r}.",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help=f"W&B entity/team. Defaults to {DEFAULT_ENTITY!r}.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"W&B project. Defaults to {DEFAULT_PROJECT!r}.",
    )
    parser.add_argument(
        "--run-path",
        default=None,
        help="Full W&B path, usually entity/project/runs/run_id. Overrides entity/project/run-id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for exported files. Defaults to {str(DEFAULT_OUTPUT_DIR)!r}.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to wandb_<run_id>.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Optional metric keys to export. Defaults to all history keys.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on history rows for quick previews.",
    )
    return parser.parse_args()


def resolve_run_path(args: argparse.Namespace) -> str:
    if args.run_path:
        return args.run_path
    if not args.run_id:
        raise ValueError("Provide --run-id or --run-path.")
    return f"{args.entity}/{args.project}/runs/{args.run_id}"


def safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    serializable = {key: safe_json_value(value) for key, value in data.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
        f.write("\n")


def collect_history(run: Any, keys: list[str] | None, max_rows: int | None) -> list[dict[str, Any]]:
    rows = []
    iterator = (
        run.scan_history(keys=keys, page_size=1000) if keys else run.scan_history(page_size=1000)
    )
    for row in iterator:
        rows.append(dict(row))
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        key for key in ("_step", "trainer/global_step", "global_step", "step") if key in fieldnames
    ]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe_json_value(value) for key, value in row.items()})


def run_id_from_path(run_path: str) -> str:
    return run_path.rstrip("/").split("/")[-1]


def main() -> int:
    args = parse_args()

    try:
        run_path = resolve_run_path(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or run_id_from_path(run_path)
    prefix = args.prefix or f"wandb_{run_id}"

    print(f"Fetching W&B run: {run_path}")
    try:
        api = wandb.Api()
        run = api.run(run_path)
    except wandb.errors.CommError as exc:
        print("Error: could not access the W&B run.")
        print(exc)
        print(
            "Check that you are logged in with `wandb login` and that the "
            "entity/project/run id are correct."
        )
        return 1

    history_rows = collect_history(run, args.keys, args.max_rows)

    history_path = output_dir / f"{prefix}_history.csv"
    summary_path = output_dir / f"{prefix}_summary.json"
    config_path = output_dir / f"{prefix}_config.json"
    metadata_path = output_dir / f"{prefix}_metadata.json"

    write_history_csv(history_path, history_rows)
    write_json(summary_path, dict(run.summary))
    write_json(config_path, dict(run.config))
    write_json(
        metadata_path,
        {
            "id": run.id,
            "name": run.name,
            "path": run_path,
            "url": run.url,
            "state": run.state,
            "created_at": run.created_at,
            "heartbeat_at": getattr(run, "heartbeatAt", None),
            "exported_at": datetime.now(UTC).isoformat(),
            "history_rows": len(history_rows),
            "history_keys_filter": args.keys,
        },
    )

    print("Export complete:")
    print(f"  history:  {history_path}")
    print(f"  summary:  {summary_path}")
    print(f"  config:   {config_path}")
    print(f"  metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
