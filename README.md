# A8 Multi-Agent Systems and Agentic AI
## Part I: GRPO Finetuning 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This repository contains the shared code and experiment records for the GRPO finetuning assignment on
`google/gemma-3-1b-it` with GSM8K for team **Dakolo**.

**Method:** We finetune `gemma-3-1b-it` with LoRA adapters using **GRPO** (Group Relative
Policy Optimisation) on GSM8K maths word problems. Rewards are fully programmatic (checks
on output format and the final numeric answer), with no human labels or learned reward model.
For each prompt, a group of `G` completions is sampled and each completion's advantage is its
reward z-scored within the group, with a KL penalty anchoring the policy to the base model.
Baseline defaults (`scripts/config.py`): group size `NUM_GENERATIONS = 2`, KL coefficient
`BETA = 0.08`, clip range `EPSILON = 0.2`, LoRA rank 64 / alpha 64, learning rate 3e-6, and
3,364 training steps on a TPU v6e-1. Each experiment branch varies one of these axes
(group size, KL budget, learning rate, LoRA capacity, or reward shaping) against this baseline.
See [scripts/README.md](scripts/README.md) for a full tour of the algorithm and codebase.
 
## Repository Structure

| Path | Purpose |
|---|---|
| `scripts/` | Training, evaluation, reward, model, data, config, and chat scripts. |
| `runs/` | Per-run metadata, logs, config snapshots, and run notes. |
| `analysis/` | Scripts for exporting metrics, plotting, and uncertainty estimates. |
| `report_assets/` | Final plots, tables, and small artefacts used in the report. |
| `docs/` | Shared planning, experiment tracking, setup notes, and patch history. |
| `tests/` | Smoke tests (`tests/smoke-tests/run_smoke_tests.py`). |
| `wandb_scripts/` | Utilities for checking and exporting W&B run data. |
| `bootstrap.sh` / `create_tpu_env.sh` | Scripts to set up the venv and TPU VM environment. |
| `tpu-setup.md` | TPU environment and setup instructions. |
| `tunix.ipynb` | Original notebook reference. |

All training runs are logged to Weights & Biases: [wandb.ai/felsomoye-university-of-cambridge/tunix](https://wandb.ai/felsomoye-university-of-cambridge/tunix).

## Branches and Experiments

Each experiment lives on its own branch. 

| Branch | Experiment / purpose |
|---|---|
| `main` | Shared code and merged experiment records. |
| `baseline-fls` | Baseline GRPO run (`NUM_GENERATIONS = 2`, run `jgs4c6kl`) and base-model eval. |
| `n-generations-4` | Group-size sweep: `NUM_GENERATIONS = 4`. |
| `n-generations-8` / `n-generations-8-rerun` | Group-size sweep: `NUM_GENERATIONS = 8` (original run and rerun). |
| `n-generations-16` | Group-size sweep: `NUM_GENERATIONS = 16`. |
| `kl-control-bk` | KL-budget experiment: `BETA = 1e-6` and `BETA = 0.32` vs. the 0.08 baseline. |
| `learning-rate-fls` | Learning-rate variation on the G=8 base. |
| `lora-rank128-alpha128-fls` | LoRA capacity: rank 128 / alpha 128. |
| `reward-length-bk` | Reward/length variant: length penalty in the reward. |
| `reward-length-on-g8-bk` | Reward/length variant run on the G=8, batch-size-1 base (incl. `g8_bs1` control). |
| `reward-reweight` | Reward reweighting variant. |
| `empty-penalty-bk` | `empty_penalty` reward to discourage empty completions. |
| `improvement-rd` | Environment fixes (e.g. pinning protobuf for the TFDS baseline). |

## Baseline Results

The baseline GRPO run (`baseline_seed42`, W&B run [`jgs4c6kl`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl), 3,364 steps in ~4.7 h on a v6e-1) learned early and then collapsed: eval reward mean peaked at **1.711 at step 448** before falling to **−0.240** by the final eval, with a KL spike to ~41 and completions degenerating to zero length.

On the 64-example greedy GSM8K eval, the final LoRA checkpoint was substantially worse than the base model:

| Model | Checkpoint | Correct | Accuracy | Partial accuracy | Format accuracy |
|---|---|---:|---:|---:|---:|
| Base `gemma-3-1b-it` (no restored LoRA) | n/a | 33/64 | 51.56% | 53.12% | 6.25% |
| LoRA-finetuned baseline | step 3364 | 2/64 | 3.12% | 6.25% | 12.50% |

Retained-checkpoint evals show progressive degradation rather than a one-off bad final copy:

| Checkpoint | Accuracy |
|---|---:|
| Base `gemma-3-1b-it` (no LoRA) | 51.56% |
| step 2000 | 28.12% |
| step 2500 | 20.31% |
| step 3000 | 6.25% |
| step 3364 (final) | 3.12% |

See [docs/I1_BASELINE_RESULTS.md](docs/I1_BASELINE_RESULTS.md) for the full record (run metadata, eval logs, training curves, and the one accepted baseline patch).

## Access the TPU

```bash
gcloud auth login

export TEAM=dakolo

gcloud alpha compute tpus tpu-vm ssh $TEAM \
  --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap

cd tpu-2026
```

## Run Bootstrapping

Compute 95% bootstrap confidence intervals for the GSM8K metrics (accuracy, partial
accuracy, format accuracy) for both the **base** `gemma-3-1b-it` and a **fine-tuned LoRA**
checkpoint, over the full 1319-question test split. Method: empirical percentile bootstrap,
10,000 iterations, seeded for reproducibility (see [docs/BOOTSTRAP.md](docs/BOOTSTRAP.md)).

The eval step needs the TPU `tunix` venv and `wandb login`; the bootstrap step itself is
plain numpy. Run from the repo root:

```bash
# Interactive — prompts for a run label and the W&B checkpoint URL:
./scripts/run_bootstrap.sh

# Or pass them as arguments to skip the prompts:
./scripts/run_bootstrap.sh k8 https://wandb.ai/felsomoye-university-of-cambridge/tunix/artifacts/model/8k-baseline-6516-steps-rd-actor-ckpt
```

The script is **idempotent**: it evaluates the base model and the fine-tuned LoRA only if
their per-question `.jsonl` files are missing, then bootstraps both. Delete a `.jsonl` (or set
`FORCE_EVAL=1`) to force a fresh eval. Other env overrides: `N_ITER` (default 10000),
`SEED` (default 42), `NUM_TEST_BATCHES` (default 1319), `VENV`.

### Where results are stored

Everything lands in `analysis/` (for a run labelled `k8`):

| File | Contents |
|---|---|
| `analysis/base_no_ft.jsonl` | Per-question results for the base model (1319 lines; reused across runs). |
| `analysis/k8_lora.jsonl` | Per-question results for the fine-tuned LoRA checkpoint (1319 lines). |
| `analysis/bootstrap_results_k8.txt` | Human-readable summary: one table per model (base first, fine-tuned second) with point value, 95% CI, bootstrap mean, and std error for accuracy / partial / format. |

The summary `.txt` is also printed to the terminal at the end of the run.

To re-bootstrap from existing `.jsonl` files without touching the TPU (e.g. to change `N_ITER`),
call the bootstrap step directly:

```bash
python scripts/bootstrap.py ci analysis/k8_lora.jsonl --label "fine-tuned LoRA (k8)" \
  --n-iter 10000 --seed 42 --output analysis/bootstrap_results_k8.txt
```

## Team Members
Barbara Koch, Funmi Looi-Somoye, Rowan d’Auria

**University of Cambridge**

## Acknowledgements

The original code was forked from [Boris Bolliet](https://github.com/borisbolliet)'s
[tpu-2026](https://github.com/borisbolliet/tpu-2026) repository, at commit
[`324abbe`](https://github.com/borisbolliet/tpu-2026/commit/324abbe4b4e229ea812223856393547db4fbb53e)
