#!/usr/bin/env python3
"""
WandB Run Status Checker
-----------------------
This script queries the Weights & Biases (WandB) API to fetch and display:
- Run metadata (name, state, URL, duration, creation time)
- Logged configuration hyperparameters
- Latest training summary metrics (e.g., loss, reward, step count)

Usage:
    python check_wandb.py --run-id <run_id>
    python check_wandb.py --run-id <run_id> --entity my-team --project my-project
"""

import argparse
import sys

DEFAULT_ENTITY = "felsomoye-university-of-cambridge"
DEFAULT_PROJECT = "tunix"

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed in the active environment.")
    print(
        "Please activate your virtual environment, for example "
        "`source venv/bin/activate` or `source ~/venvs/tunix/bin/activate`, "
        "and try again."
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and display W&B run metadata, config, and summary metrics."
    )
    parser.add_argument("--run-id", required=True, help="W&B run id, for example jgs4c6kl.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help=f"W&B entity/team. Defaults to {DEFAULT_ENTITY!r}.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"W&B project. Defaults to {DEFAULT_PROJECT!r}.")
    return parser.parse_args()


def format_value(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main():
    args = parse_args()
    run_path = f"{args.entity}/{args.project}/runs/{args.run_id}"

    print(f"Connecting to WandB run: {run_path}...")
    try:
        api = wandb.Api()
        run = api.run(run_path)

        print("\n" + "=" * 40)
        print(" RUN METADATA")
        print("=" * 40)
        print(f"Name:        {run.name}")
        print(f"State:       {run.state.upper()}")
        print(f"URL:         {run.url}")
        print(f"Created At:  {run.created_at}")
        print(f"Heartbeat:   {run.heartbeatAt}")
        print(f"Duration:    {run.summary.get('_runtime', 'N/A')} seconds")

        print("\n" + "=" * 40)
        print(" CONFIGURATION PARAMETERS")
        print("=" * 40)
        config = run.config
        if config:
            for k, v in sorted(config.items()):
                if not k.startswith("_"):
                    print(f"  {k}: {v}")
        else:
            print("  No configuration parameters logged.")

        print("\n" + "=" * 40)
        print(" LATEST SUMMARY METRICS")
        print("=" * 40)
        summary = run.summary
        if summary:
            for k, v in sorted(summary.items()):
                if not k.startswith("_"):
                    print(f"  {k}: {format_value(v)}")
        else:
            print("  No metrics recorded in summary yet.")

    except wandb.errors.CommError as e:
        print("\n[!] Communication / Authentication Error:")
        print(e)
        print("\nTo authenticate, please run:")
        print("  wandb login")
        print(
            "or set the WANDB_API_KEY environment variable in your terminal "
            "before running this script."
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
