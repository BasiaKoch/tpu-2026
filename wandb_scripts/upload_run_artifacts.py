#!/usr/bin/env python3
"""Upload a completed run's checkpoint and TensorBoard logs to W&B."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tensorboard", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    return parser.parse_args()


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} directory does not exist: {path}")
    if not any(path.rglob("*")):
        raise SystemExit(f"{label} directory is empty: {path}")


def main() -> int:
    args = parse_args()
    require_directory(args.checkpoint, "Checkpoint")
    require_directory(args.tensorboard, "TensorBoard")

    metadata = {
        "source_run_id": args.run_id,
        "git_commit": args.commit,
        "seed": args.seed,
        "step": args.step,
    }

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="must",
    )
    try:
        checkpoint = wandb.Artifact(
            name=f"{args.artifact_prefix}_checkpoint",
            type="model",
            description=f"Final checkpoint at step {args.step}.",
            metadata=metadata,
        )
        checkpoint.add_dir(str(args.checkpoint), name=f"actor/{args.step}")
        checkpoint_artifact = run.log_artifact(
            checkpoint,
            aliases=["latest", "final", f"step-{args.step}", f"seed-{args.seed}"],
        )
        checkpoint_artifact.wait()

        tensorboard = wandb.Artifact(
            name=f"{args.artifact_prefix}_tensorboard",
            type="tensorboard",
            description="TensorBoard event logs from the completed training run.",
            metadata=metadata,
        )
        tensorboard.add_dir(str(args.tensorboard), name="tensorboard")
        tensorboard_artifact = run.log_artifact(
            tensorboard,
            aliases=["latest", "final", f"seed-{args.seed}"],
        )
        tensorboard_artifact.wait()

        print(f"checkpoint={checkpoint_artifact.qualified_name}")
        print(f"checkpoint_version={checkpoint_artifact.version}")
        print(f"tensorboard={tensorboard_artifact.qualified_name}")
        print(f"tensorboard_version={tensorboard_artifact.version}")
    finally:
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
