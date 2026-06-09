#!/usr/bin/env python3
"""
WandB Run Status Checker
-----------------------
This script queries the Weights & Biases (WandB) API to fetch and display:
- Run metadata (name, state, URL, duration, creation time)
- Logged configuration hyperparameters
- Latest training summary metrics (e.g., loss, reward, step count)

To check a different run, edit the `WANDB_RUN_PATH` constant below.
"""

import sys

# ==============================================================================
# CONFIGURATION: Change your WandB run path here
# ==============================================================================
# The format should be "entity/project/runs/run_id"
RUN_ID = "jgs4c6kl"
WANDB_RUN_PATH = f"felsomoye-university-of-cambridge/tunix/runs/{RUN_ID}"
# ==============================================================================

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


def format_value(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main():
    print(f"Connecting to WandB run: {WANDB_RUN_PATH}...")
    try:
        api = wandb.Api()
        run = api.run(WANDB_RUN_PATH)

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
