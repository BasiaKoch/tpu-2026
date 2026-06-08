# Part I (I.1–I.3) — Group Project Operational Plan

Shared team playbook for the GRPO-finetuning practical (sections I.1–I.3, marked out of
50/120): reproducing the baseline, understanding it, and running a controlled improvement
experiment on `google/gemma-3-1b-it` / GSM8K, on a shared `v6e-1` TPU during the
**Mon 8 June 2026 → Mon 15 June 2026** window.

The binding constraint is wall-clock: a full baseline run takes ~5–9h, so the window fits
only a handful of complete runs. **Treat every full TPU run as a logged scientific
experiment, not a casual test.**

Note on authorship: runs, code, logs, and the TPU are *shared*; the report write-up is
*individual* — each member writes their own prose from the shared artefacts (may emphasise
different runs/plots/interpretations of the same data, but must not submit identical prose).

---

## 1. Non-negotiable rules

- [ ] No full TPU run from uncommitted code.
- [ ] No full TPU run before a small smoke test passes.
- [ ] No uncontrolled comparison: change one main thing at a time; same data, same eval
      split, same total compute (or normalised per step) across baseline and variants.
- [ ] Every run gets logged with: commit hash, config snapshot, exact command, seed, logs,
      checkpoint path, evaluation result, short interpretation.
- [ ] GitHub may be used for collaboration; the **submitted** repository must be **GitLab**,
      with all changes committed and all logs included or linked.
- [ ] Every external URL in the report must be a clickable hyperlink.
- [ ] Shared code/logs/checkpoints/plots: fine. Shared report **prose**: not allowed.
- [ ] Every team member must be able to explain any line of the submitted code in the viva.

---

## 2. Team roles (fill in immediately)

| Role | Person | Responsibilities |
|---|---|---|
| Repo owner | | Maintains GitHub fork, creates/owns the GitLab repo, branch hygiene, checks run metadata is committed |
| TPU operator | | Runs setup, starts/stops jobs, uses tmux, monitors training, records logs & checkpoint paths |
| Experiment designer | | Chooses variant(s), writes pre-run hypotheses, checks comparisons stay controlled |
| Code reviewer | | Reviews every config/code diff before a full run; prevents accidental multi-variable changes |
| Analysis owner | | Builds plots, comparison table, bootstrap CI / second-seed uncertainty, report assets |

Regardless of role, **everyone** must read and understand (needed for I.2 and the viva):
`scripts/train.py`, `scripts/rewards.py`, `scripts/config.py`, `scripts/evaluate.py`,
`scripts/data.py`, `scripts/model.py`, `scripts/chat.py`, `tunix.ipynb`.

---

## 3. Immediate next actions (before any real TPU training)

- [ ] Confirm all teammates have GitHub collaborator access; each pushes a tiny test branch
      to prove write access.
- [ ] Create working branch `baseline-fls`.
- [ ] Add tracking files: `RUNS.md`, `TEAM_PLAN.md`, `SETUP_NOTES.md`, `EXPERIMENTS.md`,
      `BASELINE_PATCHES.md` (templates in §6).
- [ ] Add folders: `runs/`, `analysis/`, `report_assets/` (layout in §5).
- [ ] Read `tpu-setup.md`, `bootstrap.sh`, `create_tpu_env.sh`, `requirements.txt`.
- [ ] Decide W&B project name + run-naming convention (e.g.
      `2026-06-08_baseline_seed42`, `2026-06-09_reward_length_penalty_seed42`).
- [ ] Agree on the first runs to target: baseline, + two variants (see §11).
- [ ] **Create the GitLab repo now**, even while GitHub remains the active collaboration repo
      — don't leave migration to the end.
- [ ] Prepare a smoke-test config/run before launching anything full-scale (§8.3).
- [ ] Read LoRA background (Appendix A of the exam, then Hu et al. 2021,
      https://arxiv.org/abs/2106.09685) and review GRPO theory from lectures — needed to
      motivate I.3 and to answer I.2/I.4.

---

## 4. GitHub / GitLab workflow

```bash
# Clone your fork, add upstream
git clone https://github.com/<you>/tpu-2026.git && cd tpu-2026
git remote add upstream https://github.com/borisbolliet/tpu-2026.git && git fetch upstream

# Working branch
git checkout -b baseline-fls
git push -u origin baseline-fls
```

Branching strategy:
```text
main                     # stays stable
baseline-fls
reward-variant
kl-control-variant
analysis-plots
```
- Full runs launch from a **committed** branch (`git status` clean).
- Merge via PR where possible; don't push directly to `main` without team agreement.

GitLab (the actual submission target):
```bash
git remote add gitlab <YOUR_GITLAB_REPO_URL>
git push gitlab main
git push gitlab --all
```
By the end, GitLab must contain the final code and either contain or link **all** logs.

---

## 5. Recommended repo layout

```text
tpu-2026/
├── scripts/                 # train.py, evaluate.py, rewards.py, config.py, data.py, model.py, chat.py
├── runs/
│   ├── 2026-06-08_base_eval_seed42/
│   ├── 2026-06-08_baseline_seed42/
│   ├── 2026-06-09_reward_length_penalty_seed42/
│   └── 2026-06-10_kl_beta_low_seed42/
├── analysis/
│   ├── export_wandb_metrics.py
│   ├── make_plots.py
│   └── bootstrap_eval_ci.py
├── report_assets/
│   ├── baseline_reward_curve.pdf
│   ├── baseline_kl_curve.pdf
│   ├── all_runs_reward.pdf
│   ├── all_runs_kl.pdf
│   ├── diagnostic_response_length.pdf
│   └── results_table.csv
├── RUNS.md  TEAM_PLAN.md  EXPERIMENTS.md  SETUP_NOTES.md  BASELINE_PATCHES.md
└── README.md
```

**Commit:** code/config diffs, run metadata, small logs, small CSV/JSON metrics, analysis
scripts, final plots, report assets, notes.
**Be careful with (link instead of commit if large):** checkpoints, full W&B dirs, large
TensorBoard event files, model weights — push to W&B artifacts / cloud storage and link
from both the GitLab repo and the report.

---

## 6. Tracking-file templates

### 6.1 `TEAM_PLAN.md`
```markdown
# Team Plan
## Members
- ...

## Roles
| Role | Person |
|---|---|
| Repo owner | |
| TPU operator | |
| Experiment designer | |
| Code reviewer | |
| Analysis owner | |

## Target full runs
1. Base model evaluation
2. Baseline GRPO
3. Variant A (reward/length)
4. Variant B (KL control)
5. Optional: second seed of best variant

## Rules
- No full TPU run from uncommitted code or before a smoke test.
- Every run logged: commit, config, command, seed, logs, checkpoint, eval.
- Shared data OK; shared report prose is not.
```

### 6.2 `RUNS.md` (one row per run — single source of truth for write-ups)
```markdown
| Run | Commit | Branch | Config | Seed | Steps | Wall-clock | Checkpoint | Eval split | Eval acc | Log link | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|---:|---|---|---|
| base_eval_seed42 | | baseline-fls | default | 42 | n/a | | n/a | GSM8K test | | | planned | base model only |
| baseline_seed42 | | baseline-fls | default | 42 | | | | GSM8K test | | | planned | default GRPO |
| reward_length_penalty_seed42 | | reward-variant | | 42 | | | | GSM8K test | | | planned | reward shaping |
| kl_beta_low_seed42 | | kl-control-variant | | 42 | | | | GSM8K test | | | planned | KL-control test |
```

### 6.3 Per-run notes (`runs/<RUN_NAME>/NOTES.md`)
```markdown
# Run: <RUN_NAME>
## Purpose / Hypothesis
What question does this answer? Theory predicts X because Y; expect Z in reward/KL/length/accuracy.
## Code state
Branch / commit / `git status` clean (yes/no)
## Config
File, changed parameters, seed, data split, total steps
## Command
```bash
<exact command>
```
## Timing
Start / end / wall-clock
## Outputs
W&B URL, TensorBoard URL, train log, eval log, checkpoint path, eval accuracy
## Observations
Reward behaviour, KL behaviour, instability, hardware/logging issues
## Interpretation
Did this match the hypothesis? Why / why not?
```

---

## 7. Run-metadata commands (use for every full run, no exceptions)

```bash
# Pre-run: clean commit + snapshot
git status && git add . && git commit -m "Prepare <run-name>" && git rev-parse HEAD

RUN_NAME=2026-06-08_baseline_seed42
mkdir -p runs/$RUN_NAME
git rev-parse HEAD            > runs/$RUN_NAME/git_commit.txt
git status                    > runs/$RUN_NAME/git_status_before.txt
cp scripts/config.py            runs/$RUN_NAME/config_snapshot.py
echo "python scripts/train.py" > runs/$RUN_NAME/command.txt
date                          > runs/$RUN_NAME/start_time.txt

# Launch with full log capture
python scripts/train.py 2>&1 | tee runs/$RUN_NAME/train.log

# Post-run
date                          > runs/$RUN_NAME/end_time.txt
python scripts/evaluate.py 2>&1 | tee runs/$RUN_NAME/eval.log
git status                    > runs/$RUN_NAME/git_status_after.txt

# Commit metadata
git add runs/$RUN_NAME RUNS.md EXPERIMENTS.md SETUP_NOTES.md BASELINE_PATCHES.md
git commit -m "Add metadata for $RUN_NAME" && git push
```

---

## 8. TPU workflow

### 8.1 First login & sanity
Follow the course setup guide exactly; verify `pwd`, `ls`, `python --version`,
`git --version`. Clone the repo on the TPU, checkout `baseline-fls`, run setup
per `tpu-setup.md`, and **log every deviation in `SETUP_NOTES.md`**.

### 8.2 tmux discipline
```bash
tmux new -s baseline      # start
# Ctrl-b d                # detach
tmux attach -t baseline   # reattach
tmux ls                   # list
```
Always run training inside tmux — losing an SSH connection must not kill a 5–9h job.

### 8.3 Smoke test before any full run
A smoke test (`debug_smoke_test_do_not_report/`) must prove:
- [ ] model loads, dataset loads
- [ ] a few training steps run without error
- [ ] W&B / local logging works
- [ ] checkpoint path is valid and writable
- [ ] evaluation script runs (on a tiny subset if supported)

**Never use smoke-test numbers as experimental evidence** — they're for setup validation only.
Document any setup patch discovered here. Only after this passes, launch the full baseline.

---

## 9. I.1 — Reproduce the baseline (10 marks)

### Pre-run checklist
- [ ] On `baseline-fls`, `git status` clean, **default** `scripts/config.py` restored
- [ ] W&B project configured, run name agreed, seed fixed
- [ ] Everyone knows this is the *official* baseline run

### Launch & evaluate (see §7 for the metadata commands)
- [ ] Run `scripts/train.py` end-to-end with default config; capture commit hash, wall-clock,
      step count, W&B/log URL, checkpoint path
- [ ] Document any as-shipped failure + fix in `BASELINE_PATCHES.md` (single shared section
      every member can summarise individually)
- [ ] Run `scripts/evaluate.py` on:
  - [ ] (a) the **base** model `gemma-3-1b-it` (no adapter)
  - [ ] (b) the **LoRA-finetuned** checkpoint
  — same held-out GSM8K split, same fixed seed, both recorded in `RUNS.md`
- [ ] Export: mean-reward-vs-step curve, KL(π_θ‖π_ref)-vs-step curve

### Definition of done
- [ ] Training completed (or failure fully documented) — wall-clock & step count recorded
- [ ] W&B/log URL + checkpoint path saved
- [ ] Both base and LoRA accuracies recorded with split name + seed
- [ ] Reward curve & KL curve data exported to `report_assets/`
- [ ] `BASELINE_PATCHES.md` updated if anything needed patching

---

## 10. I.2 — Understand the baseline (10 marks)

Hold **one team walkthrough** so everyone arrives at correct answers, then each person
writes independent prose (≤ 1 page). Checklist to resolve together:

- [ ] What plays π_θ, π_old, π_ref in the LoRA setup? Which carry trainable parameters?
      (Appendix A: all three share frozen W₀ and differ only by adapter snapshot; π_ref is
      the B = 0 base — costs nothing to keep around for the KL penalty.)
- [ ] What is group size `K` in `scripts/config.py`? Does Tunix use the standard group-mean
      advantage or leave-one-out? Trace it in the Tunix source, don't guess.
- [ ] In `scripts/rewards.py`, which of the 4 terms are *shaping* rewards, which is the
      *true correctness* reward? What happens early in training with correctness-only reward,
      in terms of within-group variance σ_r (degenerate groups ⇒ zero advantage signal)?
- [ ] Where is the clipped PPO-style surrogate implemented in Tunix? Which `config.py` field
      is ε? Any deviation from the lecture form (e.g. per-token vs. per-sequence ratio)?

---

## 11. I.3 — Improve on the baseline (30 marks)

Target: **≥ 3 complete runs** total (baseline + 2 more). Controlled comparison: same data,
same eval split, same total compute (or per-step normalised). Pick **one main change** per
variant — multi-variable changes are uninterpretable and will be rejected by the code reviewer.

### 11.1 Variant A — reward shaping / length penalty (recommended)
| | |
|---|---|
| Change | Edit `scripts/rewards.py`: add a mild length penalty, reduce/remove an approximate-format term, or tighten format matching to exact |
| Hypothesis | Shaping helps early when correctness is sparse, but excess shaping rewards format/verbosity over correctness; a mild length penalty may curb response-length blowup |
| Expected reward/KL | Mean reward may dip if shaping is reduced; KL shouldn't move sharply unless behaviour changes a lot |
| Expected accuracy | Could rise (less reward-hacking/verbosity) or fall (shaping was load-bearing early on) |
| Diagnostic | Response length over training; malformed-output rate; fraction of correct numeric answers |
| Risk / fallback | Too strong a penalty discourages reasoning → reduce weight or use strict-format-only |

### 11.2 Variant B — KL control (recommended)
| | |
|---|---|
| Change | One conservative change to β or ε in `scripts/config.py` (e.g. lower β or slightly wider ε) |
| Hypothesis | Weaker KL control ⇒ faster but riskier policy movement; stronger control ⇒ stable but possibly underfit |
| Expected reward/KL | Lower β / wider ε ⇒ faster reward rise + more KL drift; tighter control ⇒ opposite |
| Expected accuracy | Best accuracy likely at a *moderate* KL budget, not the extremes |
| Diagnostic | KL(π_θ‖π_ref) vs. accuracy, plotted alongside baseline |
| Risk / fallback | Too weak control ⇒ instability/reward hacking → use a smaller Δβ/Δε or revert |

### 11.3 Variant C — LoRA rank (optional, only if time allows)
| | |
|---|---|
| Change | Raise/lower LoRA rank `r` in `scripts/config.py` |
| Hypothesis | Higher rank ⇒ more adapter capacity but faster KL-budget consumption / possible overfit; lower rank ⇒ more regularised but possibly underfit |
| Diagnostic | KL vs. accuracy; reward curve; final eval accuracy |
| Note | Less directly tied to GRPO mechanics — deprioritise vs. A/B |

**Avoid unless everything else is done:** mixing in a second dataset, large K sweeps,
rewriting the advantage estimator, multiple simultaneous hyperparameter changes,
architecture-level changes.

### Required report artefacts (collect into `report_assets/`)
- [ ] (a) Comparison table — base / baseline / improved checkpoint(s), with seeds + uncertainty
      (second seed or bootstrap CI over eval prompts)
- [ ] (b) Combined training curves — mean reward & KL, baseline + all variants, one set of axes
- [ ] (c) One diagnostic plot for a known GRPO failure mode — pick to match your variant:
      Â_i distribution / fraction of degenerate groups (all K rewards equal) / response
      length & entropy over time / KL vs. accuracy
- [ ] (d) Notes connecting findings to lecture theory, **including any contradicted prediction**
      (honest negative results are explicitly in scope and valued for diagnosis quality)

---

## 12. Day-by-day schedule

| Day | Goal | Definition of done |
|---|---|---|
| Mon 8 Jun | Setup + smoke test | Smoke test passes, `SETUP_NOTES.md` committed |
| Tue 9 Jun | Full baseline | Baseline run logged, evaluated (base + LoRA), pushed |
| Wed 10 Jun | Code walkthrough + Variant A debug | I.2 answers agreed; Variant A code reviewed & smoke-tested |
| Thu 11 Jun | Full Variant A | Run complete, evaluated, logged, interpreted |
| Fri 12 Jun | Full Variant B | KL-control variant complete, evaluated, logged |
| Sat 13 Jun | Analysis | Comparison table, combined curves, diagnostic plot, CI — all in `report_assets/` |
| Sun 14 Jun | Buffer | Rerun a failed variant or 2nd seed; verify GitLab mirror is current |
| Mon 15 Jun | Window closes | No new risky training; final sync/backup; everything accessible |

**Minimum viable outcome** if things go badly: base eval + full baseline + **one** completed
principled variant + one partial/failed variant with a clearly documented incident +
controlled comparison and honest discussion. Target remains baseline + 2 full variants.

---

## 13. Monitoring & kill criteria

**Watch:** mean reward, KL to reference, loss, step time, memory/device errors, NaNs,
checkpoint saves, response length (if logged), W&B logging health.

**Kill a run early only if:** loss/reward goes NaN; KL explodes immediately and clearly
invalidates the run; checkpointing is broken; logging is unrecoverably broken; wrong
config/branch/uncommitted code was launched; device errors make output unusable.

**Do not kill** just because reward looks noisy early — GRPO is noisy by nature.

---

## 14. Analysis & uncertainty plan

Generate everything via scripts in `analysis/` (not manual screenshots), saving to
`report_assets/`: `export_wandb_metrics.py` → `make_plots.py` → `bootstrap_eval_ci.py`.

Per-run data to export: `step`, `mean_reward`, `kl`, `loss`, `learning_rate`,
`response_length_mean`, `entropy` (if logged), eval accuracy, per-question correctness
(needed for bootstrap CI).

**Bootstrap CI** (if only one seed per variant): resample eval prompts with replacement,
compute accuracy, repeat ≥ 1000×, report mean + 95% CI. If per-question predictions aren't
available, say so honestly and prioritise a second seed for the most important variant
instead.

---

## 15. Risk register

| Risk | Mitigation | Recovery |
|---|---|---|
| TPU setup fails | Start immediately, log every error | Ask course staff; `SETUP_NOTES.md`; fewer-but-cleaner runs |
| Baseline fails as-shipped | Smoke test first | `BASELINE_PATCHES.md`; patch only env/path issues |
| W&B fails | Also `tee` to local logs | Fall back to TensorBoard/JSONL/local logs |
| Checkpoint not saved | Verify path in smoke test | Use last available checkpoint; document loss |
| Run launched from dirty/wrong code | Require clean `git status` + commit hash before every run | Mark run invalid; exclude from main evidence |
| Variant changes too many things | Code review before launch | Revert; run the simpler single-variable version |
| KL explodes | Conservative β/ε deltas | Kill if clearly invalid; rerun with safer setting |
| Variant underperforms | Write hypothesis *before* the run | Honest negative result + theory discussion (explicitly in scope) |
| Fewer than 3 runs | Prioritise baseline + 2 simple variants | Document TPU failures/partial runs explicitly |
| Teammates overwrite each other's code | Branches + PRs | Restore from git history |
| GitLab migration forgotten | Create GitLab repo on day 1 | Push all branches/logs before the deadline |
| Shared prose accidentally copied | Shared *artefacts* only, never shared write-ups | Each member writes independent interpretation |

---

## 16. Submission-readiness checklist

- [ ] Final code lives in **GitLab**; all logs included or linked from it
- [ ] Every external URL (GitLab repo, W&B runs, dashboards, dataset cards, papers) is a
      clickable hyperlink (`\href{}{}` / `\url{}`)
- [ ] Every reported number traces to a logged run in `RUNS.md`; every figure traces to
      saved data + a script in `analysis/`
- [ ] Team members listed explicitly on the title page (so the marker can identify
      jointly-owned experiments)
- [ ] Each member has written their own independent prose from the shared artefacts
- [ ] I.1–I.3 write-up ≤ 3 pages (A4, 11pt, ≥ 2cm margins, single column, single spacing,
      typeset in LaTeX as part of the single coherent report)
- [ ] Everyone can explain any submitted line of code on request (viva)

---

## 17. Priority ranking

**Today:** confirm access → tracking files & folders → create GitLab repo → read setup
files → W&B naming convention → prepare smoke test → agree baseline + two safe variants.

**Can wait:** plot styling, bootstrap CI script polish, optional second seed, LoRA-rank
experiment, report prose itself.

**Avoid during the TPU window unless everything else is done:** dataset mixing, large
hyperparameter sweeps, rewriting the GRPO learner, simultaneous multi-variable changes,
undocumented manual experiments, leaving GitLab migration to the last day.

---

## 18. Final recommended run order

| # | Run | Purpose | Priority |
|---|---|---|---|
| 0 | Smoke test | Prove the setup works | Essential |
| 1 | Base-model eval | Required baseline number | Essential |
| 2 | Default GRPO baseline | Required reproduction (I.1) | Essential |
| 3 | Variant A — reward/length | Tests reward shaping & length control | High |
| 4 | Variant B — KL control | Tests β/ε and KL drift | High |
| 5 | Second seed of best variant | Uncertainty measure, if time allows | Optional |

Safest narrative for the report: *the baseline reproduces and trains as expected; reward
shaping changes the optimisation signal and output behaviour; KL control governs policy
drift; final accuracy must be read together with reward, KL, and a diagnostic plot — not
in isolation.*
