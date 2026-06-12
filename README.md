# TPU 2026

Shared code and experiment records for the GRPO finetuning practical on
`google/gemma-3-1b-it` with GSM8K.

## Repository Map

| Path | Purpose |
|---|---|
| `scripts/` | Training, evaluation, reward, model, data, config, and chat scripts. |
| `runs/` | Per-run metadata, logs, config snapshots, and run notes. |
| `analysis/` | Scripts for exporting metrics, plotting, and uncertainty estimates. |
| `report_assets/` | Final plots, tables, and small artefacts used in the report. |
| `docs/` | Shared planning, experiment tracking, setup notes, and patch history. |
| `GROUP_PROJECT_PLAN.md` | Operational plan for the group project. |
| `tpu-setup.md` | TPU environment and setup instructions. |
| `tunix.ipynb` | Original notebook reference. |

## Core Docs

- [Team plan](docs/TEAM_PLAN.md)
- [Run log](docs/RUNS.md)
- [Experiment plan](docs/EXPERIMENTS.md)
- [Setup notes](docs/SETUP_NOTES.md)
- [Baseline patches](docs/BASELINE_PATCHES.md)
- [Project plan](GROUP_PROJECT_PLAN.md)
- [TPU setup](tpu-setup.md)
- [Scripts guide](scripts/README.md)

## Access the TPU

```bash
gcloud auth login

export TEAM=dakolo

gcloud alpha compute tpus tpu-vm ssh $TEAM \
  --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap

cd tpu-2026
```

## Run Discipline

- No full TPU run from uncommitted code.
- No full TPU run before a smoke test passes.
- Every run should record commit hash, config, command, seed, logs, checkpoint, evaluation result, and interpretation.
- Shared code, logs, plots, and metrics are fine; report prose should be written independently.

