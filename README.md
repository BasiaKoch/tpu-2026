# A8 — GRPO Finetuning of Gemma 3 on GSM8K

GRPO (Group Relative Policy Optimisation) finetune of
[`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) on the
[GSM8K](https://arxiv.org/abs/2110.14168) maths word-problem benchmark, built on
[Tunix](https://github.com/google/tunix) / JAX with LoRA adapters and a fully
programmatic (non-learned) reward.

This is the GitLab submission repository for **Rowan d'Auria** (`rd761`). Sections I.1–I.3 of
the report describe a **shared team project** (team DAKOLO): the TPU, the runs, and this code
are jointly owned. Section I.4 and Part II are individual.

**Team members** (report title page): Rowan d'Auria, Basia Koch, Funmi Looi-Somoye —
University of Cambridge.

The code was developed on the team's GitHub fork
([`borisbolliet/tpu-2026`](https://github.com/borisbolliet/tpu-2026) → team fork) and ported
here in full: **this repository is the submission, and every experiment branch and every
report-cited run commit lives in it** (see the branch and commit links below).

---

## For the marker — start here

- **The code lives at the repository root.** [`CODEBASE.md`](CODEBASE.md) is a full tour of the
  algorithm and codebase, and [`scripts/README.md`](scripts/README.md) walks through every
  script. This top-level README is an **index that maps the report to its evidence** (runs,
  logs, eval artifacts) so nothing has to be retyped or hunted for.
- **Report-cited paths** such as [`scripts/config.py`](scripts/config.py),
  [`scripts/rewards.py`](scripts/rewards.py), and [`evaluation/`](evaluation/) are at the
  repository root, exactly as named in the report.
- **Every cited run is pinned to an immutable commit.** Each row in the tables below links its
  exact code commit; the three headline runs are also tagged (`run/k2-baseline`,
  `run/k8-baseline`, `run/k8-new-reward`). Each experiment also has its own branch (the
  [branch → run map](#branch--run-map)).
- **Logs are linked, not lost:** every training run is on
  [Weights & Biases](https://wandb.ai/felsomoye-university-of-cambridge/tunix) (clickable run
  links in the tables below), per-question eval dumps and bootstrap CIs are committed in
  [`evaluation/`](evaluation/), and the consolidated run ledger is [`docs/RUNS.md`](docs/RUNS.md).

---

## Runs cited in the report

All runs use **seed 42** and the GSM8K test split. "64-ex" = first 64 test examples, greedy
decoding; "full-test" = all 1319 test prompts with a seeded non-parametric bootstrap (10,000
resamples) 95% CI. $K$ = `NUM_GENERATIONS`, the GRPO group size. "Commit" links the exact code
that produced the run (immutable permalink).

### Controlled comparison — report Table 1 (§I.3), full-test bootstrap

The headline result. Same data, same eval split, same 5864-step schedule; numbers come from the
per-question dumps in [`evaluation/`](evaluation/).

| Run | Modification | Commit | Steps | Accuracy | 95% CI | Evidence |
|---|---|---|---:|---:|---|---|
| Base `gemma-3-1b-it` | no fine-tuning | [`7e696c4`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/commit/7e696c4) | — | 625/1319 = 47.38% | [44.81%, 50.11%] | [`base_no_ft.jsonl`](evaluation/base_no_ft.jsonl) |
| K=2 LoRA | default reward (baseline config) | [`7e696c4`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/commit/7e696c4) | 5864 | 1/1319 = 0.08% | [0.00%, 0.23%] | [`k2_baseline_lora.jsonl`](evaluation/k2_baseline_lora.jsonl), [`bootstrap_results_k2_baseline.txt`](evaluation/bootstrap_results_k2_baseline.txt) |
| K=8 LoRA | group size $K:2\to8$, default reward | [`99059ce`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/commit/99059ce) | 5864 | 739/1319 = 56.03% | [53.45%, 58.76%] | [`k8-6516-steps_lora.jsonl`](evaluation/k8-6516-steps_lora.jsonl), [`bootstrap_results_k8-6516-steps.txt`](evaluation/bootstrap_results_k8-6516-steps.txt) |
| **K=8 new reward** | K=8 + correctness-heavy reward | [`e4f6d69`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/commit/e4f6d69) | 5864 | **772/1319 = 58.53%** | [55.88%, 61.18%] | [`k-8-new-reward`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/k-8-new-reward) · [`k8-new-reward_lora.jsonl`](evaluation/k8-new-reward_lora.jsonl), [`bootstrap_results_k8-new-reward.txt`](evaluation/bootstrap_results_k8-new-reward.txt) |

The three headline runs are also tagged for convenience:
[`run/k2-baseline`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/run/k2-baseline) (`7e696c4`),
[`run/k8-baseline`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/run/k8-baseline) (`99059ce`),
[`run/k8-new-reward`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/run/k8-new-reward) (`e4f6d69`).
Paired base-vs-finetuned bootstrap (with McNemar + Holm correction):
[`compare_all.txt`](evaluation/compare_all.txt).

### Group-size ($K$) sweep — report appendix (Table `tab:ksweep`), 64-ex greedy

Commits are the per-run code state from [`docs/RUNS.md`](docs/RUNS.md) (the single source of truth).

| $K$ | W&B run | Commit | Wall time | Steps | Accuracy | Partial | Format |
|---:|---|---|---|---:|---:|---:|---:|
| 2 | [`jgs4c6kl`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl) | [`7e696c4`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/commit/7e696c4) | 4.69 h | 3364 | 3.12% (2/64) | 6.25% | 12.50% |
| 4 | [`x4j7yhdp`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/x4j7yhdp) | [`1db3d25`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/commit/1db3d25) | 6.63 h | 3364 | 54.69% (35/64) | 59.38% | 89.06% |
| 8 | [`m3xp6k97`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/m3xp6k97) | [`99059ce`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/commit/99059ce) | 7.41 h | 3364 | 59.38% (38/64) | 62.50% | 95.31% |
| 16 | [`xqbl406c`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/xqbl406c) | [`17c90d5`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/commit/17c90d5) | 8.03 h | ≈2500 (failed @2991) | 56.25% (36/64) | 59.38% | 92.19% |

### Other single-variable runs — report appendix (Table `tab:other-runs`), 64-ex greedy, 3364 steps

| Run | Change vs baseline | $K$ | Wall | W&B run | LoRA acc (64) |
|---|---|---:|---|---|---:|
| LR 1e-5 | LR 3e-6 → 1e-5 | 8 | 8h03 | [`hozux9t6`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/hozux9t6) | 29.69% (19/64) |
| LoRA rank 128 | rank/α 64 → 128 (with K=8) | 8 | 8h32 | [`aoz8dtkp`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/aoz8dtkp) | 29.69% (19/64) |
| LoRA rank 128 | rank/α 64 → 128 | 2 | 5h29 | [`v5cvlwkm`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/v5cvlwkm) | 32.81% (21/64) |
| β = 1e-6 | β 0.08 → 1e-6 (KL ≈ off) | 2 | 3h26 | [`8rmv0hgg`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/8rmv0hgg) | 51.56% (33/64) |
| β = 0.32 | β 0.08 → 0.32 (4× leash) | 2 | 1h07 | [`oet2tfjd`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/oet2tfjd) | 0.00% (0/64) |
| Length penalty | + length penalty | 2 | 2h34 | [`jcp0b5cy`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jcp0b5cy) | 4.69% (3/64) |
| Length penalty | + length penalty (batch 2 → 1) | 8 | 5h17 | [`cyay16mj`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/cyay16mj) | 50.00% (32/64) |
| Hard/medium data | hard/medium curriculum (with K=8) | 8 | 7h36 | [`6yiowy1y`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/6yiowy1y) | 46.88% (30/64) |
| Empty penalty | + empty-completion penalty | 2 | — | not logged | 14.06% (9/64) |

> §I.1 baseline run identity: commit
> [`7e696c4`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/commit/7e696c4),
> branch [`baseline-fls`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/baseline-fls),
> W&B run [`jgs4c6kl`](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl)
> (4h41m, 3364 GRPO steps). The only baseline patch was pinning `protobuf==6.31.1` for
> TFDS/GSM8K loading — see [`docs/BASELINE_PATCHES.md`](docs/BASELINE_PATCHES.md).

The single source of truth for every run (completed, failed, planned), with commits, W&B ids,
checkpoints, and notes, is [`docs/RUNS.md`](docs/RUNS.md).

---

## Branch → run map

Every experiment is on its own branch **in this repository** (ported from the team fork); the
report's headline runs come from three of them. Branch names link to the branch in GitLab; see
[`docs/RUNS.md`](docs/RUNS.md) for the full per-branch ledger.

| Branch | Produced (report) |
|---|---|
| [`main`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/main) | Shared code and merged experiment records (this submission). |
| [`n-generations-8`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/n-generations-8) | **K=8 baseline** (`m3xp6k97`, 59.38% 64-ex) — the K=8 LoRA row of Table 1 and the K=8 row of the sweep; reference point for every K=8 single-change run. |
| [`reward-reweight`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/reward-reweight) | **K=8 new reward** (`k-8-new-reward`, 58.53% full-test / 71.88% 64-ex) — the best result, "K=8 new reward" in Table 1. |
| [`baseline-fls`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/baseline-fls) | K=2 baseline (`jgs4c6kl`) and the base-model eval (§I.1). |
| [`n-generations-4`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/n-generations-4) / [`n-generations-16`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/n-generations-16) | K-sweep points (`x4j7yhdp`, `xqbl406c`). |
| [`kl-control-bk`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/kl-control-bk) | β sweep (`8rmv0hgg`, `oet2tfjd`). |
| [`learning-rate-fls`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/learning-rate-fls) | LR 1e-5 (`hozux9t6`). |
| [`lora-rank128-alpha128-fls`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/lora-rank128-alpha128-fls) | LoRA rank 128 (`aoz8dtkp`, `v5cvlwkm`). |
| [`reward-length-bk`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/reward-length-bk) / [`reward-length-on-g8-bk`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/reward-length-on-g8-bk) | Length-penalty runs (`jcp0b5cy`, `cyay16mj`). |
| [`empty-penalty-bk`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/empty-penalty-bk) | Empty-completion penalty (14.06%). |
| [`medium-hard-data-fls`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/medium-hard-data-fls) | Hard/medium curriculum (`6yiowy1y`). |
| [`analysis-bk`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/analysis-bk) | Eval dumps, bootstrap CIs, training-curve overlays, degenerate-group diagnostic. |

---

## Where the logs live

- **Weights & Biases** — all training/eval runs:
  [`felsomoye-university-of-cambridge/tunix`](https://wandb.ai/felsomoye-university-of-cambridge/tunix)
  (per-run links in the tables above). Summary dashboard of all experiments:
  [GRPO Experiments report](https://api.wandb.ai/links/felsomoye-university-of-cambridge/c2r3q1wi).
- **Per-question eval dumps + bootstrap CIs** — [`evaluation/`](evaluation/)
  (`*.jsonl` per-question results, `bootstrap_results_*.txt` CI summaries, `compare_all.txt`
  paired comparison).
- **Run ledger and setup notes** — [`docs/`](docs/)
  ([`RUNS.md`](docs/RUNS.md), [`EXPERIMENTS.md`](docs/EXPERIMENTS.md),
  [`BASELINE_RESULTS.md`](docs/BASELINE_RESULTS.md),
  [`BASELINE_PATCHES.md`](docs/BASELINE_PATCHES.md),
  [`SETUP_NOTES.md`](docs/SETUP_NOTES.md)).
- **Per-run training folders** (config snapshots, logs) — [`runs/`](runs/).

---

## Reproduce

All commands run from the repository root on a TPU **v6e-1** VM. Full setup, including the
one-time `python3.12` / TPU steps, is in [`tpu-setup.md`](tpu-setup.md) and
[`bootstrap.sh`](bootstrap.sh).

**Setup** (creates `~/venvs/tunix` and installs the pinned JAX/Tunix/LoRA stack):

```bash
./bootstrap.sh                       # python3.12 venv + requirements.txt + jax/tunix/qwix/flax
# or, with uv:
uv venv ~/venvs/tunix --python 3.12 && source ~/venvs/tunix/bin/activate
uv pip install -r requirements.txt
```

**Train** (GRPO; runs under tmux so the run survives a dropped SSH session):

```bash
./scripts/run_tmux.sh                # cd scripts && python train.py, in a detached tmux session
# or directly:
cd scripts && python train.py
```

**Key knobs.** Hyperparameters live in [`scripts/config.py`](scripts/config.py). Per-run
overrides are read from the environment — captured in the W&B config snapshot rather than edited
in code:

| Env var | Default | Controls |
|---|---|---|
| `SEED` | 42 | Data order + train/val split. |
| `BETA` | 0.08 | KL penalty coefficient β (e.g. `BETA=1e-6` to switch the leash off). |
| `DATA_SOURCE` | `tfds` | GSM8K source (`tfds` or `kaggle`). |
| `TPU_CONTENT_DIR` | `/tmp/content` | Checkpoint/tensorboard root (point at persistent storage for long runs). |
| `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_RUN_ID` | `tunix` / … / none | W&B logging target; set `WANDB_RUN_ID` to resume a run. |

The group size `NUM_GENERATIONS` (K), `NUM_BATCHES`, LoRA `RANK`/`ALPHA`, and `LEARNING_RATE`
are set in [`scripts/config.py`](scripts/config.py) and varied **per experiment branch** (e.g.
`n-generations-8` sets `NUM_GENERATIONS = 8`).

**Evaluate** a checkpoint on the GSM8K test split:

```bash
cd scripts
python evaluate.py --ckpt-dir <ckpt-root> --step <N>   # greedy, 64-example by default
python evaluate.py --no-restore                        # base model (no LoRA) reference
```

**Bootstrap CIs + paired comparison** over the full 1319-example test set (writes into
[`evaluation/`](evaluation/)):

```bash
./scripts/run_bootstrap.sh k8 <wandb-checkpoint-url>            # eval + 10k-iter bootstrap CI
CHECKPOINT_PATH=~/checkpoints/k8/actor/5864 ./scripts/run_bootstrap.sh k8   # from a local ckpt
python scripts/bootstrap.py compare evaluation/base_no_ft.jsonl \
  evaluation/k8-new-reward_lora.jsonl --labels "base,k8+reward"   # paired diff + McNemar/Holm
```

---

## External references (clickable)

- Base model — [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it)
- Dataset — GSM8K: [paper (Cobbe et al. 2021)](https://arxiv.org/abs/2110.14168) ·
  [TFDS catalog](https://www.tensorflow.org/datasets/catalog/gsm8k) ·
  [Kaggle mirror](https://www.kaggle.com/datasets/thedevastator/grade-school-math-8k-q-a)
- LoRA — [Hu et al. 2021](https://arxiv.org/abs/2106.09685) (Appendix A of the report)
- Training framework — [Tunix](https://github.com/google/tunix)
- Upstream baseline — forked from [borisbolliet/tpu-2026](https://github.com/borisbolliet/tpu-2026)
  at commit [`324abbe`](https://github.com/borisbolliet/tpu-2026/commit/324abbe4b4e229ea812223856393547db4fbb53e)
- Part II — `cmbagent_lg` adaptive-planning, exact commit studied:
  [`d7d0592`](https://github.com/borisbolliet/cmbagent_lg/commit/d7d0592a714e4cc01c97f1b77afdd57a208b18db)
  ([branch tree](https://github.com/borisbolliet/cmbagent_lg/tree/d7d0592a714e4cc01c97f1b77afdd57a208b18db))

---

## Acknowledgements

We thank Dr Boris Bolliet for setting up the team's Google Cloud TPU and providing the starter
code for this assignment.
