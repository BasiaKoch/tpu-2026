# Team Plan

## Members

- ...

## Roles

| Role | Person | Responsibilities |
|---|---|---|
| Repo owner | | Maintains repo hygiene, GitLab mirror, branches, and run metadata checks. |
| TPU operator | | Runs setup, launches jobs, monitors tmux/W&B/logs, records checkpoint paths. |
| Experiment designer | | Defines hypotheses and keeps comparisons controlled. |
| Code reviewer | | Reviews config/code diffs before full runs. |
| Analysis owner | | Builds plots, tables, confidence intervals, and report assets. |

## Target Full Runs

1. Base model evaluation.
2. Baseline GRPO.
3. Variant A: reward shaping or length penalty.
4. Variant B: KL control.
5. Optional: second seed of the best variant.

## Rules

- No full TPU run from uncommitted code or before a smoke test.
- Every run logs commit, config, command, seed, logs, checkpoint, and evaluation.
- Shared data and artefacts are OK; shared report prose is not.
- Every team member should be able to explain the submitted code.

