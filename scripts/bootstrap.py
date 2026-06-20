#!/usr/bin/env python3
"""Bootstrap CIs for GSM8K accuracy — self-contained eval + resampling.

This file deliberately duplicates the evaluation pipeline from `scripts/evaluate.py`
so that the bootstrap workflow lives entirely here and `scripts/evaluate.py` stays
untouched. It has two subcommands:

  eval  Generate the per-question results that the bootstrap consumes. Loads the
        model, optionally restores a fine-tuned LoRA checkpoint downloaded from a
        W&B artifact (URL or ref), evaluates over the GSM8K test split, and writes
        one {question, answer, correct, partial, format} JSON record per line.
        Requires the TPU `tunix` venv (jax / tunix / wandb).

  ci    Empirical percentile bootstrap over a .jsonl for all three 0/1 metrics
        (accuracy/`correct`, `partial`, `format`): point value, 95% CI, bootstrap
        mean + std error each, computed off one shared set of resample indices so
        the metrics are mutually coherent. Pure numpy — runs anywhere. Seeded
        (default 42) for reproducibility. The CI captures sampling uncertainty
        over the test set, not generation randomness (eval uses deterministic
        greedy decoding). These are per-model (marginal) CIs — to compare two
        models use `compare`, not whether these CIs overlap (overlap ignores the
        cross-model covariance and overstates a difference's uncertainty).

  compare  Paired bootstrap across two or more per-question .jsonls covering the
        same split: reports each pairwise accuracy difference with a 95% CI and a
        two-sided bootstrap p-value, all off ONE shared set of resample indices so
        the difference carries the cross-model covariance the per-model `ci` CIs
        throw away. Cross-checked with the exact McNemar test and Holm-corrected
        across the comparisons. Guards that the files are row-for-row aligned (same
        questions, same order) before pairing. Pure numpy.

Usage:
  python bootstrap.py eval --no-restore --dump-jsonl analysis/base_no_ft.jsonl
  python bootstrap.py eval --wandb-artifact <URL> --dump-jsonl analysis/k8_lora.jsonl
  python bootstrap.py ci analysis/k8_lora.jsonl --label "fine-tuned LoRA (k8)" \
      --n-iter 10000 --seed 42 --output analysis/bootstrap_results_k8.txt
  python bootstrap.py compare analysis/base_no_ft.jsonl analysis/k8_lora.jsonl \
      --labels "base,fine-tuned (k8)" --output analysis/compare_k8.txt
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys

# Directory holding config.py / data.py / model.py / rewards.py (the eval deps).
# This file lives in scripts/ alongside them, so it is simply its own directory.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# ci subcommand — pure-numpy bootstrap (no jax/tunix needed)
# ======================================================================

# Metrics to bootstrap, in report order. Each is a 0/1 column in the .jsonl.
METRICS = ("correct", "partial", "format")
METRIC_LABELS = {"correct": "accuracy", "partial": "partial", "format": "format"}


def load_metrics(path):
    """Read a per-question .jsonl into 0/1 metric arrays plus the question list.

    The questions are returned alongside the metrics so a paired comparison can
    verify that two files cover the same questions in the same order before it
    pairs their per-question outcomes (see `load_aligned`). Callers that only need
    the metrics can ignore the question list.
    """
    import numpy as np
    cols = {m: [] for m in METRICS}
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            questions.append(rec["question"])
            for m in METRICS:
                cols[m].append(int(rec[m]))
    n = len(cols["correct"])
    if n == 0:
        raise ValueError(f"No records found in {path!r}.")
    return {m: np.asarray(v, dtype=np.float64) for m, v in cols.items()}, questions, n


def bootstrap_metrics(cols, n, n_iter, seed):
    """Bootstrap every metric off ONE shared set of resample indices.

    Sharing the (n_iter, n) index matrix across metrics keeps the CIs mutually
    coherent (each bootstrap replicate is the same resampled set of questions
    scored under all three metrics). Returns {metric: (point, lo, hi, mean, se)}.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))      # shared across metrics
    out = {}
    for m in METRICS:
        v = cols[m]
        means = v[idx].mean(axis=1)
        point = float(v.mean())
        lo, hi = (float(x) for x in np.percentile(means, [2.5, 97.5]))
        out[m] = (point, lo, hi, float(means.mean()), float(means.std(ddof=1)))
    return out


def format_block(label, path, n, n_iter, seed, results):
    lines = [
        "=" * 72,
        label,
        "=" * 72,
        f"  source jsonl    : {path}",
        f"  n_samples       : {n}",
        f"  n_iter          : {n_iter}",
        f"  seed            : {seed}",
        "",
        f"  {'metric':<9}{'point':<18}{'95% CI':<28}{'boot mean':<12}{'std err'}",
        f"  {'-' * 9}{'-' * 18}{'-' * 28}{'-' * 12}{'-' * 8}",
    ]
    for m in METRICS:
        point, lo, hi, bmean, se = results[m]
        ci = f"[{lo * 100:.2f}%, {hi * 100:.2f}%]"
        lines.append(
            f"  {METRIC_LABELS[m]:<9}"
            f"{point * 100:.2f}% ({point:.4f}) "
            f"{ci:<28}"
            f"{bmean * 100:.2f}%{'':<6}"
            f"{se * 100:.2f}%"
        )
    return "\n".join(lines) + "\n"


def cmd_ci(args):
    cols, _, n = load_metrics(args.input)
    results = bootstrap_metrics(cols, n, args.n_iter, args.seed)
    label = args.label or args.input
    block = format_block(label, args.input, n, args.n_iter, args.seed, results)
    print(block)
    for m in METRICS:
        point, lo, hi, _, _ = results[m]
        if not (lo <= point <= hi):
            print(f"WARNING: point {METRIC_LABELS[m]} falls outside its bootstrap CI.",
                  file=sys.stderr)
    if args.output:
        with open(args.output, "a" if args.append else "w", encoding="utf-8") as f:
            f.write(block + "\n")
    return 0


# ======================================================================
# compare subcommand — paired bootstrap of pairwise accuracy differences
# ======================================================================

def load_aligned(paths, labels=None):
    """Load several per-question .jsonls and verify they are row-for-row paired.

    Pairing the per-question outcomes is only valid when row i is the SAME
    question in every file — greedy decoding gives each model exactly one outcome
    per question on the same split, so the outcomes are paired and a difference
    inherits the cross-model covariance. The eval pipeline writes the files in a
    fixed test-set order, but nothing downstream enforces it, so this guards it
    explicitly: every file must have the same length and the same question text at
    every position, otherwise the comparison would silently pair unrelated
    questions. Returns (labels, cols_list, n).
    """
    if len(paths) < 2:
        raise ValueError("compare needs at least two .jsonl files.")
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(labels) != len(paths):
        raise ValueError(f"got {len(labels)} labels for {len(paths)} inputs.")
    cols_list, ref_q, n = [], None, None
    for path in paths:
        cols, questions, count = load_metrics(path)
        if ref_q is None:
            ref_q, n = questions, count
        elif count != n:
            raise ValueError(
                f"{path!r} has {count} questions but {paths[0]!r} has {n}; the "
                f"files must cover the same test split to be paired.")
        elif questions != ref_q:
            bad = next(i for i in range(n) if questions[i] != ref_q[i])
            raise ValueError(
                f"{path!r} disagrees with {paths[0]!r} at row {bad} "
                f"({questions[bad]!r} vs {ref_q[bad]!r}); the files are not "
                f"row-for-row aligned and cannot be paired.")
        cols_list.append(cols)
    return labels, cols_list, n


def holm_adjust(pvals):
    """Holm step-down adjusted p-values, returned in the input order.

    Controls the family-wise error rate across the pairwise comparisons: sort the
    m raw p-values ascending, scale the k-th smallest by (m - k), and enforce
    monotonicity. Compare the adjusted values against a single alpha (e.g. 0.05).
    Pure python — no scipy.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def mcnemar_exact_p(a, b):
    """Two-sided exact McNemar p-value for two 0/1 outcome arrays on the same items.

    The classic test for paired binary outcomes: it conditions on the discordant
    pairs (questions where exactly one model is correct) and asks whether they
    split evenly. Under H0 the count favouring either model is Binomial(d, 0.5),
    so the exact two-sided p doubles the smaller tail. Serves as a closed-form
    cross-check on the paired bootstrap. Pure python (math.comb) — works on numpy
    arrays or plain lists.
    """
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    d = n01 + n10
    if d == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(comb(d, i) for i in range(k + 1)) / (2 ** d)
    return min(2.0 * tail, 1.0)


def bootstrap_compare(cols_list, n, n_iter, seed, metric="correct"):
    """Paired bootstrap of every pairwise accuracy difference.

    All models are scored on ONE shared (n_iter, n) set of resample indices, so
    each replicate scores the same resampled questions under every model and the
    per-replicate difference carries the cross-model covariance. Because per-
    question correctness is positively correlated across these models, that makes
    a difference's CI tighter than bootstrapping each model on its own indices
    (the independent-CI-overlap method, which overstates the difference's
    uncertainty). Returns (points, pairs) where points[i] is model i's accuracy
    and pairs is a list of dicts: {i, j, diff, lo, hi, p_boot, p_mcnemar}.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))          # shared across ALL models
    vals = [c[metric] for c in cols_list]
    points = [float(v.mean()) for v in vals]
    accs = [v[idx].mean(axis=1) for v in vals]           # (n_iter,) per model
    pairs = []
    for i, j in itertools.combinations(range(len(cols_list)), 2):
        d = accs[j] - accs[i]                             # paired per-replicate diff
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        # Two-sided bootstrap p: twice the smaller mass on one side of zero.
        p_boot = min(1.0, 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean())))
        pairs.append({
            "i": i, "j": j, "diff": points[j] - points[i], "lo": lo, "hi": hi,
            "p_boot": p_boot, "p_mcnemar": mcnemar_exact_p(vals[i], vals[j]),
        })
    return points, pairs


def _fmt_p(p):
    """Compact p-value: '<1e-4' below the bootstrap resolution, else 4 dp."""
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def _stars(p):
    """Significance marker for a (Holm-adjusted) p-value."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def format_compare_block(labels, paths, n, n_iter, seed, metric, points, pairs):
    metric_label = METRIC_LABELS.get(metric, metric)
    holm = holm_adjust([pr["p_mcnemar"] for pr in pairs])
    comps = [f"{labels[pr['j']]} - {labels[pr['i']]}" for pr in pairs]
    wlbl = max(len(s) for s in labels)
    wcomp = max(len(c) for c in comps)
    lines = [
        "=" * 72,
        f"Paired bootstrap comparison — {metric_label}",
        "=" * 72,
        f"  n_samples       : {n}",
        f"  n_iter          : {n_iter}",
        f"  seed            : {seed}",
        f"  comparisons     : {len(pairs)} (Holm-corrected, family-wise alpha = 0.05)",
        "",
        "  sources:",
    ]
    for lbl, path in zip(labels, paths):
        lines.append(f"    {lbl:<{wlbl}}  {path}")
    lines += ["", "  per-model accuracy (point):"]
    for lbl, pt in zip(labels, points):
        lines.append(f"    {lbl:<{wlbl}}  {pt * 100:6.2f}%")
    lines += [
        "",
        "  pairwise differences (paired bootstrap; 95% CI per comparison):",
        f"    {'comparison':<{wcomp}}  {'diff':>8}  {'95% CI':<22}"
        f"  {'boot p':>8}  {'McNemar p':>10}  {'Holm p':>8}  sig",
        f"    {'-' * wcomp}  {'-' * 8}  {'-' * 22}  {'-' * 8}  {'-' * 10}  {'-' * 8}  ---",
    ]
    for comp, pr, hp in zip(comps, pairs, holm):
        ci = f"[{pr['lo'] * 100:+.2f}%, {pr['hi'] * 100:+.2f}%]"
        lines.append(
            f"    {comp:<{wcomp}}  {pr['diff'] * 100:+7.2f}%  {ci:<22}"
            f"  {_fmt_p(pr['p_boot']):>8}  {_fmt_p(pr['p_mcnemar']):>10}"
            f"  {_fmt_p(hp):>8}  {_stars(hp)}"
        )
    lines += [
        "",
        "  diff = acc(B) − acc(A) for 'B - A'; a 95% CI excluding 0 means the pair",
        "  differs at ~5% for that single comparison. 'Holm p' / sig control the",
        "  family-wise error across all pairwise tests via Holm on the exact McNemar",
        "  p; *** <.001  ** <.01  * <.05.",
    ]
    return "\n".join(lines) + "\n"


def cmd_compare(args):
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else None
    labels, cols_list, n = load_aligned(args.inputs, labels)
    points, pairs = bootstrap_compare(cols_list, n, args.n_iter, args.seed, args.metric)
    block = format_compare_block(
        labels, args.inputs, n, args.n_iter, args.seed, args.metric, points, pairs)
    print(block)
    if args.output:
        with open(args.output, "a" if args.append else "w", encoding="utf-8") as f:
            f.write(block + "\n")
    return 0


# ======================================================================
# eval subcommand — duplicated GSM8K eval pipeline + W&B checkpoint loading
# ======================================================================

def wandb_ref_from_url(ref):
    """Normalise a W&B artifact URL or ref into an `entity/project/name:version` ref.

    Accepts an already-valid ref (defaulting to `:latest` if no version is given)
    or a browser URL like
    `https://wandb.ai/<entity>/<project>/artifacts/model/<name>[/<version>]`.
    """
    ref = ref.strip()
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/artifacts/[^/]+/([^/?#]+)(?:/([^/?#]+))?", ref)
    if m:
        entity, project, name, version = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"{entity}/{project}/{name}:{version or 'latest'}"
    return ref if ":" in ref.split("/")[-1] else f"{ref}:latest"


def download_wandb_artifact(ref):
    """Download a W&B model artifact and return the local download directory."""
    try:
        import wandb
    except ImportError:
        raise RuntimeError(
            "wandb is not installed in this environment. "
            "Activate the TPU venv (e.g. `source ~/venvs/tunix/bin/activate`)."
        )
    full_ref = wandb_ref_from_url(ref)
    print(f"Downloading W&B artifact: {full_ref}")
    artifact = wandb.Api().artifact(full_ref, type="model")
    path = artifact.download()
    print(f"Artifact downloaded to {path}")
    return path


def _stage_bare_checkpoint(ckpt_dir, step):
    """Expose a bare step checkpoint as a CheckpointManager root.

    `ckpt_dir` holds a single step's contents (`_CHECKPOINT_METADATA` + items),
    but `CheckpointManager(root_directory=R).maybe_restore(step=S)` reads from
    `R/S/`. Build a sibling root `<ckpt_dir>__as_step/` with a `<step>` symlink
    pointing back at `ckpt_dir`, and return `(root, step)`.
    """
    root = ckpt_dir.rstrip("/") + "__as_step"
    os.makedirs(root, exist_ok=True)
    link = os.path.join(root, str(step))
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(ckpt_dir), link)
    print(f"Staged bare checkpoint as root={root} step={step}")
    return root, step


def find_ckpt_root_and_step(download_dir):
    """Locate the orbax checkpoint root + step inside a downloaded artifact.

    `CheckpointManager(root_directory=R)` restores step S from `R/S/...`, so we
    look for a directory whose children include integer-named (step) subdirs.
    Prefer a path mentioning `actor` (the policy checkpoint) when several exist.
    """
    candidates = []
    for dirpath, dirnames, _ in os.walk(download_dir):
        steps = [int(d) for d in dirnames if d.isdigit()]
        if steps:
            candidates.append((dirpath, max(steps)))
    if not candidates:
        # Some artifacts are uploaded as the *contents* of a single step dir
        # (orbax writes `_CHECKPOINT_METADATA` at the step root) rather than the
        # CheckpointManager root that wraps it. Stage it as `<root>/<step>/` via a
        # symlink so CheckpointManager can restore it. The step value is arbitrary
        # here (restore only uses it to locate the dir), so use a placeholder.
        if os.path.exists(os.path.join(download_dir, "_CHECKPOINT_METADATA")):
            base = os.path.basename(download_dir.rstrip("/"))
            if base.isdigit():
                # Already orbax-shaped (e.g. `.../actor/5864`): the parent is the
                # CheckpointManager root and this dir's name is the step.
                root = os.path.dirname(os.path.abspath(download_dir.rstrip("/")))
                step = int(base)
                print(f"Detected checkpoint root={root} step={step}")
                return root, step
            # Non-zero step: maybe_restore returns the step on success and the
            # caller treats a 0 return as "nothing restored".
            return _stage_bare_checkpoint(download_dir, step=1)
        raise RuntimeError(
            f"No orbax step subdirs found under {download_dir!r}; "
            f"cannot determine the checkpoint root automatically."
        )
    actor = [c for c in candidates if "actor" in c[0].lower()]
    root, step = (actor or candidates)[0]
    print(f"Auto-detected checkpoint root={root} step={step}")
    return root, step


def cmd_eval(args):
    # Make the scripts/ modules importable and resolve relative data dirs the same
    # way scripts/evaluate.py does (it runs from scripts/). Output path is made
    # absolute first so the chdir doesn't move it.
    dump_path = os.path.abspath(args.dump_jsonl)
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    os.chdir(SCRIPTS_DIR)

    from tqdm.auto import tqdm
    from tunix.generate import sampler as sampler_lib
    from tunix.sft.checkpoint_manager import CheckpointManager

    from config import (
        GENERATION_CONFIGS, MAX_PROMPT_LENGTH, TEST_DATA_DIR, TOTAL_GENERATION_STEPS,
        TRAIN_DATA_DIR, TRAIN_FRACTION, TRAIN_MICRO_BATCH_SIZE, NUM_BATCHES, NUM_EPOCHS,
        DATA_SOURCE,
    )
    from data import SYSTEM_PROMPT, TEMPLATE, build_train_val_test
    from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer
    from rewards import match_format, match_numbers

    def generate(questions, sampler, eos_tokens, temperature, top_k, top_p, seed=None):
        batch = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q) for q in questions]
        out = sampler(
            input_strings=batch, max_generation_steps=TOTAL_GENERATION_STEPS,
            temperature=temperature, top_k=top_k, top_p=top_p,
            echo=False, seed=seed, eos_tokens=eos_tokens,
        )
        return out.text

    def evaluate(dataset, sampler, eos_tokens, temperature=0.7, top_k=50, top_p=0.95, num_passes=1):
        corr = partially_corr = corr_format = total = 0
        records = []
        for batch in tqdm(dataset):
            answers = batch["answer"]
            questions = batch["question"]
            per_q = [[] for _ in range(len(questions))]
            for p in range(num_passes):
                responses = generate(questions, sampler, eos_tokens, temperature, top_k, top_p, seed=p)
                for i, r in enumerate(responses):
                    per_q[i].append(r)
            for q, responses, ans in zip(questions, per_q, answers):
                got_corr = got_partial = got_format = False
                for r in responses:
                    ext = guess.group(1) if (guess := match_numbers.search(r)) is not None else "-1e9"
                    try:
                        if float(ext.strip()) == float(ans.strip()):
                            got_corr = True
                        ratio = float(ext.strip()) / float(ans.strip())
                        if 0.9 <= ratio <= 1.1:
                            got_partial = True
                    except Exception:
                        pass
                    if match_format.search(r) is not None:
                        got_format = True
                    if got_corr and got_partial and got_format:
                        break
                corr += int(got_corr)
                partially_corr += int(got_partial)
                corr_format += int(got_format)
                total += 1
                records.append({
                    "question": q, "answer": ans,
                    "correct": int(got_corr), "partial": int(got_partial), "format": int(got_format),
                })
                if total % 10 == 0:
                    print(f"===> corr={corr} total={total} acc={corr/total*100:.2f}% "
                          f"partial={partially_corr/total*100:.2f}% fmt={corr_format/total*100:.2f}%")
        return corr, total, records

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    if args.no_restore:
        print("Skipping checkpoint restore; evaluating base model / zero-init LoRA adapter.")
    elif args.wandb_artifact or args.checkpoint_path:
        if args.checkpoint_path:
            ckpt_dir = os.path.abspath(os.path.expanduser(args.checkpoint_path))
            if not os.path.isdir(ckpt_dir):
                raise SystemExit(f"--checkpoint-path not found: {ckpt_dir}")
        else:
            ckpt_dir = download_wandb_artifact(args.wandb_artifact)
        ckpt_root, step = find_ckpt_root_and_step(ckpt_dir)
        mgr = CheckpointManager(root_directory=ckpt_root)
        n, _ = mgr.maybe_restore(model=lora, step=step, restore_only_lora_params=True)
        if n == 0:
            raise RuntimeError(f"No checkpoint restored from {ckpt_root} (step {step}).")
        print(f"Restored LoRA params from step {n}")
    else:
        raise SystemExit("eval requires --no-restore, --wandb-artifact, or --checkpoint-path.")

    _, _, test_ds = build_train_val_test(
        NUM_BATCHES, args.num_test_batches, TRAIN_MICRO_BATCH_SIZE, TRAIN_FRACTION,
        NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR, source=args.source,
    )

    sampler = sampler_lib.Sampler(
        transformer=lora, tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
        ),
    )
    n, t, records = evaluate(test_ds, sampler, eos_tokens, **GENERATION_CONFIGS[args.preset])

    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nFINAL: correct={n}/{t}  acc={n/t*100:.2f}%")
    print(f"Wrote {len(records)} per-question records to {dump_path}")
    return 0


# ======================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Generate per-question .jsonl (needs the TPU venv).")
    pe.add_argument("--dump-jsonl", required=True, help="Output .jsonl path.")
    pe.add_argument("--no-restore", action="store_true",
                    help="Evaluate the base model / zero-init LoRA adapter (no fine-tuning).")
    pe.add_argument("--wandb-artifact", default=None,
                    help="W&B model artifact URL or ref to download and restore.")
    pe.add_argument("--checkpoint-path", default=None,
                    help="Local checkpoint dir to restore (e.g. .../actor/5864 or its "
                         "CheckpointManager root). Skips the W&B download.")
    pe.add_argument("--preset", default="greedy", help="Generation preset (default greedy).")
    pe.add_argument("--source", default="tfds", choices=["tfds", "kaggle"])
    pe.add_argument("--num-test-batches", type=int, default=1319,
                    help="Test questions to eval (TRAIN_MICRO_BATCH_SIZE=1). Default 1319 = full split.")
    pe.set_defaults(func=cmd_eval)

    pc = sub.add_parser("ci", help="Percentile bootstrap over a .jsonl (pure numpy).")
    pc.add_argument("input", help="Per-question .jsonl with a 0/1 `correct` field.")
    pc.add_argument("--n-iter", type=int, default=10000, help="Bootstrap iterations (default 10000).")
    pc.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default 42).")
    pc.add_argument("--label", default=None, help="Human-readable label for this model.")
    pc.add_argument("--output", default=None, help="Write the summary block to this .txt.")
    pc.add_argument("--append", action="store_true", help="Append to --output instead of overwriting.")
    pc.set_defaults(func=cmd_ci)

    px = sub.add_parser("compare",
                        help="Paired bootstrap of pairwise accuracy differences across models.")
    px.add_argument("inputs", nargs="+",
                    help="Two or more per-question .jsonls covering the same test split.")
    px.add_argument("--labels", default=None,
                    help="Comma-separated model labels, one per input (default: file stems).")
    px.add_argument("--metric", default="correct", choices=list(METRICS),
                    help="Which 0/1 metric to compare (default correct = accuracy).")
    px.add_argument("--n-iter", type=int, default=10000, help="Bootstrap iterations (default 10000).")
    px.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default 42).")
    px.add_argument("--output", default=None, help="Write the summary block to this .txt.")
    px.add_argument("--append", action="store_true", help="Append to --output instead of overwriting.")
    px.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
