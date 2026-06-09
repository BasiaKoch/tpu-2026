# Run Recording Checklist

Follow this for **every** experiment so all 7 of us produce comparable, reproducible
records. It mirrors the layout of `runs/2026-06-08_baseline_seed42/`. When in doubt,
copy that folder and overwrite the contents.

## 0. Before you start

- [ ] Branch off `main` with a descriptive name (e.g. `reward-variant`, `kl-control-variant`).
- [ ] **Keep `seed=42`** unless your experiment *is* a seed study — this is what makes runs comparable.
- [ ] Change **one variable at a time** vs. the baseline. Note exactly what it is.
- [ ] Add a `planned` row to `docs/RUNS.md` so others know it's in flight.

## 1. Create the run folder

Name: `runs/YYYY-MM-DD_<shortname>_seed<NN>/` (e.g. `runs/2026-06-10_reward_length_penalty_seed42/`).

## 2. Files every run folder MUST contain

Reproducibility / provenance:
- [ ] `command.txt` — exact launcher + training command used.
- [ ] `config_snapshot.py` — copy of the config you actually ran (not a link to a mutable file).
- [ ] `git_commit.txt` — full commit SHA the run was launched from.
- [ ] `git_status_before.txt` and `git_status_after.txt` — uncommitted diff state at launch/finish.
- [ ] `start_time.txt` and `end_time.txt` — UTC ISO timestamps.
- [ ] `status.txt` — one of `planned` / `running` / `completed` / `failed` / `invalid`.

Outcome:
- [ ] `completion_summary.txt` — run, commit, start/end, wall-clock, final step, exit code, checkpoint path, log markers.
- [ ] `notes.md` — what you changed vs. baseline, *why*, what happened, and the next required action / interpretation.
- [ ] `checkpoint_dir.txt` — path to the final checkpoint + any preserved copy / W&B artifact.

Tracking links:
- [ ] `wandb_url.txt` — W&B run URL (and checkpoint-artifact URL if uploaded).
- [ ] `tensorboard.txt` — TensorBoard logdir / link if used.

Evaluation:
- [ ] `eval_summary.txt` — see eval section below.
- [ ] `base_eval.log` and `lora_eval.log` — raw eval output (plus any checkpoint-sweep logs, e.g. `lora_eval_step3000.log`).

If it failed:
- [ ] `failure_summary.txt` describing where/why it died; set `status.txt=failed`; move the folder to `..._failed/` if superseded.

## 3. Weights & Biases (W&B) — everyone uploads the same way

So results and checkpoints are comparable and recoverable across all 7 of us.

Identity (set these on every run):
- [ ] Same entity/project for everyone: `felsomoye-university-of-cambridge/tunix`. Do **not** log to personal projects.
- [ ] Run name = your `runs/` folder name minus the date (e.g. `reward_length_penalty_seed42`).
- [ ] `group` = experiment family; `job_type` = `train` / `eval` / `upload` so the run types stay linked.
- [ ] Tags on every run: branch name, `seed42`, the one variable studied (e.g. `reward-variant`), and `baseline` vs `experiment`.
- [ ] Confirm W&B captured the git commit (launch from inside the repo; commit must not show as dirty/unknown).

Checkpoints (log as versioned artifacts):
- [ ] Upload the **final checkpoint** as artifact `<run>_checkpoint`, `type="model"` (baseline: `baseline_seed42_checkpoint:latest`).
- [ ] Also upload the **eval-sweep checkpoints** (the same steps you eval — e.g. 2000/2500/3000/final) as versions of that artifact.
      Final + sweep steps only — do **not** upload every step (storage quota).
- [ ] Put the **commit SHA and seed** in the artifact metadata/aliases so a checkpoint is never ambiguous.

Metrics & files (so the W&B run is self-contained):
- [ ] Log the four eval metrics (`acc`, `partial`, `format`, `correct/64`) for **both base and restored checkpoint** as W&B summary metrics — not just in the log file — so they sort/compare in the runs table.
- [ ] Stream training curves (reward, KL, loss) live during the run.
- [ ] Attach `config_snapshot.py` and the raw eval logs (`base_eval.log`, `lora_eval.log`) as an artifact.

Cross-link:
- [ ] Record both the run URL **and** the checkpoint-artifact URL in `wandb_url.txt`.
- [ ] Point the `docs/RUNS.md` log link at the W&B run.

## 4. The variables to hold constant (so comparison is fair)

Record these in `config_snapshot.py`; only the one you're studying should differ from baseline:
- [ ] Base model + checkpoint
- [ ] Dataset + split (GSM8K)
- [ ] Seed (42)
- [ ] LoRA config (rank, alpha, target modules)
- [ ] Optimizer, learning rate, batch / micro-batch size
- [ ] Total steps / epochs
- [ ] Reward function definition
- [ ] KL coefficient (beta) and KL target
- [ ] Sampling/generation params (temp, top-p, max tokens)

State in `notes.md`: **"Changed vs. baseline: \<X>. Everything else held constant."**

## 5. Evaluation — use the EXACT same protocol as baseline

This is the most important part for fair comparison. Match the baseline `eval_summary.txt`:
- [ ] Preset: **greedy** decoding
- [ ] Split: GSM8K test, `NUM_TEST_BATCHES=64`, `TRAIN_MICRO_BATCH_SIZE=1`
- [ ] Report **all four** metrics: `correct=N/64`, `acc=%`, `partial=%`, `format=%`
- [ ] Eval **both** the base/zero-init LoRA wrapper AND your restored checkpoint (sanity check the restore).
- [ ] If you save intermediate checkpoints, run the checkpoint sweep (e.g. steps 2000/2500/3000/final) so we can see the training curve, not just the endpoint.

> Note: the baseline LoRA checkpoint *underperformed* its base model (3.12% vs 51.56%) and that's
> flagged as possibly a restore/eval bug. Eval base + checkpoint every time so you can catch the
> same issue rather than reporting a misleading number.

## 6. Finish up

- [ ] Update the run's row in `docs/RUNS.md` with: run, commit, branch, config, seed, steps,
      wall-clock, checkpoint, eval split, eval acc, log link, status, notes.
- [ ] Commit the `runs/<your-folder>/` and the `docs/RUNS.md` row on your branch.
- [ ] Open a PR (or merge per team process) so everyone sees the result.

## Quick copy-paste setup

```bash
SRC=runs/2026-06-08_baseline_seed42
DST=runs/$(date -u +%F)_<shortname>_seed42
cp -r "$SRC" "$DST"
# then overwrite every file with YOUR run's values; don't leave baseline data behind
```
