# A8 — Multi-Agent Systems and Agentic AI (GRPO Finetuning)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Shared code and experiment records for team **DAKOLO**: finetuning `google/gemma-3-1b-it`
on GSM8K maths word problems with **GRPO** (Group Relative Policy Optimisation).

## Project Overview

We finetune `gemma-3-1b-it` with LoRA adapters using GRPO. Rewards are fully programmatic
— checks on output format and the final numeric answer, with no human labels or learned
reward model. For each prompt we sample a group of `G` completions; each completion's
advantage is its reward z-scored within the group, with a KL penalty anchoring the policy
to the base model.

Every experiment branch varies **one** axis — group size, KL budget, learning rate, LoRA
capacity, or reward shaping — against the shared baseline below
(`scripts/config.py`):

| Parameter | Baseline Default |
|---|---|
| `NUM_GENERATIONS` (group size) | 2 |
| LoRA rank / alpha | 64 / 64 |
| `BETA` (KL coefficient) | 0.08 |
| Learning rate | 3e-6 |
| `EPSILON` (clip range) | 0.2 |
| Training steps | 3,364 (TPU v6e-1) |

See [scripts/README.md](scripts/README.md) for a full tour of the algorithm and codebase.

## Repository Structure

```
.
├── scripts/                # GRPO training, eval, rewards, model, data, config, chat
│   ├── baseline_scripts/   # Frozen copy of the upstream baseline scripts
│   ├── train.py            # GRPO training loop
│   ├── evaluate.py         # GSM8K evaluation
│   ├── bootstrap.py        # Bootstrap CIs + paired model comparison
│   ├── rewards.py          # Programmatic reward functions
│   ├── model.py            # Model / LoRA setup
│   ├── data.py             # GSM8K data loading
│   ├── config.py           # Hyperparameters and defaults
│   ├── chat.py             # Interactive chat with a checkpoint
│   ├── run_bootstrap.sh    # End-to-end bootstrap eval driver
│   ├── run_tmux.sh         # Launch training in tmux
│   └── README.md           # Algorithm and codebase tour
├── runs/                   # Per-run metadata, logs, config snapshots, and notes
├── evaluation/             # Per-question eval dumps (.jsonl) and bootstrap CI summaries
├── docs/                   # Experiment tracking, run records, setup notes, patch history
├── tests/                  # Smoke tests 
├── bootstrap.sh            # venv / environment setup
├── create_tpu_env.sh       # TPU VM environment setup
├── requirements.txt        # Python dependencies
├── tpu-setup.md            # TPU environment and setup instructions
├── tunix.ipynb             # Original notebook reference
├── LICENSE
└── README.md
```


## Weights & Biases

- **Summary report** of all experiments: [GRPO Experiments report](https://api.wandb.ai/links/felsomoye-university-of-cambridge/c2r3q1wi)
- **All training runs** are logged to the [`tunix` project](https://wandb.ai/felsomoye-university-of-cambridge/tunix).

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
| `medium-hard-data-fls` | Data/curriculum experiment: train on medium + hard GSM8K questions on the G=8 base. |
| `analysis-bk` | Analysis tooling: per-question eval dumps, bootstrap CI, training-curve overlays, and an offline advantage-signal (degenerate-group) diagnostic. |
| `improvement-rd` | Environment fixes (e.g. pinning protobuf for the TFDS baseline). |
| `run-record-fixes-bk` | Housekeeping: fix run records and checkpoint artifacts (KL β=0.32 and reward-length run metadata). |
| `pre-commit-hooks` | Tooling: pre-commit config, `pyproject.toml`, and formatting of the `scripts/` code. |

## Baseline Results

The baseline GRPO run (`baseline_seed42`, W&B run
[`jgs4c6kl`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl),
3,364 steps in ~4.7 h on a v6e-1) **learned early and then collapsed**: eval reward mean
peaked at **1.711 (step 448)** before falling to **−0.240** by the final eval, with a KL
spike to ~41 and completions degenerating to zero length.

On the 64-example greedy GSM8K eval, the final LoRA checkpoint was substantially worse than
the base model:

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

See [docs/BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md) for the full record (run metadata, eval logs, training curves, and the one accepted baseline patch).

## Access the TPU

```bash
gcloud auth login

export TEAM=dakolo

gcloud alpha compute tpus tpu-vm ssh $TEAM \
  --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap

cd tpu-2026
```

## Run Bootstrapping

Compute 95% bootstrap confidence intervals (accuracy, partial accuracy, format accuracy)
for the base `gemma-3-1b-it` and a fine-tuned LoRA checkpoint over the full 1319-question
GSM8K test split (10,000 iterations, seeded). The eval step needs the TPU `tunix` venv and
`wandb login`; the bootstrap itself is plain numpy. Run from the repo root:

```bash
# Interactive — prompts for a run label and W&B checkpoint URL:
./scripts/run_bootstrap.sh

# Pass them as arguments to skip the prompts:
./scripts/run_bootstrap.sh k8 <wandb-checkpoint-url>

# Or restore from a local checkpoint dir (no URL needed):
CHECKPOINT_PATH=~/checkpoints/k8/actor/5864 ./scripts/run_bootstrap.sh k8
```

The script is idempotent: it only re-evaluates a model if its per-question `.jsonl` is
missing (set `FORCE_EVAL=1` to force). Other overrides: `N_ITER`, `SEED`,
`NUM_TEST_BATCHES`, `VENV`.

Results land in `evaluation/` (for a run labelled `k8`): `base_no_ft.jsonl` and
`k8_lora.jsonl` hold the per-question results, and `bootstrap_results_k8.txt` is the
human-readable CI summary (also printed to the terminal). The summary ends with a
**paired** base-vs-fine-tuned comparison (see below).

### Comparing models (paired bootstrap)

Greedy decoding on the same questions makes the per-question outcomes *paired*, so test
differences with a paired bootstrap, not by checking whether per-model CIs overlap (which
ignores cross-model covariance and overstates the difference's uncertainty). The `compare`
subcommand resamples all models off **one** shared set of question indices and reports each
pairwise accuracy difference with a 95% CI and two-sided p-value, cross-checked against the
exact McNemar test and Holm-corrected across comparisons:

```bash
# Any number of aligned per-question .jsonls (same questions, same order):
python scripts/bootstrap.py compare \
  evaluation/base_no_ft.jsonl evaluation/k2_baseline_lora.jsonl \
  evaluation/k8_lora.jsonl evaluation/k8_reweight_lora.jsonl \
  --labels "base,k2,k8,k8+reweight" --output evaluation/compare_all.txt
```

A difference is significant when its 95% CI excludes 0 (per comparison) and its
Holm-adjusted p stays below 0.05 (family-wise across all pairs).

## Team Members

Barbara Koch · Funmi Looi-Somoye · Rowan d’Auria — **University of Cambridge**

## Acknowledgements

We thank Dr Boris Bolliet for setting up the team's Google Cloud TPU and for providing the
starter code for this assignment. The original code was forked from
[Dr Boris Bolliet](https://github.com/borisbolliet)'s
[tpu-2026](https://github.com/borisbolliet/tpu-2026) repository, at commit
[`324abbe`](https://github.com/borisbolliet/tpu-2026/commit/324abbe4b4e229ea812223856393547db4fbb53e).
