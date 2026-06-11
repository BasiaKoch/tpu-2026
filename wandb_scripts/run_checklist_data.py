#!/usr/bin/env python3
"""Print checklist file contents populated from the W&B API to stdout.

Fields that cannot be retrieved from the API are shown with a CANNOT_RETRIEVE
marker so you know exactly what still needs manual input.

Usage:
    python run_checklist_data.py --run-id <run_id>
    python run_checklist_data.py --run-id jgs4c6kl
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Activate the TPU environment first.")
    sys.exit(1)

DEFAULT_ENTITY = "felsomoye-university-of-cambridge"
DEFAULT_PROJECT = "tunix"

NA = "CANNOT_RETRIEVE"
POPULATED = "populated"
PARTIAL = "partial"
CANNOT = "cannot"


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print run checklist file contents from W&B.")
    p.add_argument("--run-id", required=True, help="W&B run id, e.g. jgs4c6kl.")
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    return p.parse_args()


def fetch_run(args: argparse.Namespace) -> Any:
    run_path = f"{args.entity}/{args.project}/runs/{args.run_id}"
    print(f"Fetching W&B run: {run_path}\n", file=sys.stderr)
    try:
        return wandb.Api().run(run_path)
    except wandb.errors.CommError as e:
        print(f"Error: could not access run.\n{e}", file=sys.stderr)
        sys.exit(1)


def emit(filename: str, content: str) -> None:
    bar = "─" * 60
    print(f"\n┌─ {filename} {bar[len(filename)+3:]}")
    for line in content.rstrip("\n").splitlines():
        print(f"│ {line}")
    print(f"└{'─' * 63}")


def iso(ts: str) -> str:
    return ts if (ts.endswith("Z") or "+" in ts) else ts + "Z"


def end_time(run: Any) -> str | None:
    rt = run.summary.get("_runtime")
    if rt is None:
        return None
    try:
        start = datetime.fromisoformat(run.created_at.rstrip("Z")).replace(tzinfo=UTC)
        return (start + timedelta(seconds=float(rt))).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def wall_clock(run: Any) -> str:
    rt = run.summary.get("_runtime")
    if rt is None:
        return f"{NA}: _runtime not in summary"
    h, rem = divmod(int(float(rt)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def final_step(run: Any) -> str:
    for key in ("trainer/global_step", "global_step", "_step"):
        v = run.summary.get(key)
        if v is not None:
            return str(int(v))
    return f"{NA}: step key not found in summary"


def state_label(run: Any) -> str:
    return {"finished": "completed", "failed": "failed", "crashed": "failed",
            "running": "running"}.get(run.state, run.state)


def git(run: Any) -> dict:
    return run.metadata.get("git", {})


def logged_model_artifacts(run: Any) -> list[Any]:
    artifacts: list[Any] = []
    try:
        for artifact in run.logged_artifacts():
            if artifact.type == "model":
                artifacts.append(artifact)
    except Exception:
        pass
    return artifacts


def artifact_ref(artifact: Any) -> str:
    name = str(artifact.name)
    if ":" in name:
        return f"{artifact.entity}/{artifact.project}/{name}"
    return f"{artifact.entity}/{artifact.project}/{name}:{artifact.version}"


def artifact_url(artifact: Any) -> str:
    name = str(artifact.name)
    artifact_name = name.split(":", 1)[0]
    version = str(getattr(artifact, "version", "") or "latest").lstrip(":")
    return f"https://wandb.ai/{artifact.entity}/{artifact.project}/artifacts/model/{artifact_name}/{version}"


def metadata_has_any(metadata: dict, keys: tuple[str, ...]) -> bool:
    lowered = {str(k).lower() for k in metadata}
    return any(k in lowered for k in keys)


# ── file content builders ─────────────────────────────────────────────────────

def show_wandb_url(run: Any) -> tuple[str, str]:
    artifact_lines = [artifact_url(a) for a in logged_model_artifacts(run)]
    lines = [
        "W&B run URL:",
        run.url,
        "",
        "Checkpoint artifact URL(s):",
        *(artifact_lines or [f"{NA}: no model artifact found — upload per CHECKLIST.md §3"]),
    ]
    emit("wandb_url.txt", "\n".join(lines))
    return (POPULATED if artifact_lines else PARTIAL), \
           ("run URL + artifact URL(s)" if artifact_lines else "run URL only; no model artifact URL")


def show_wandb_identity(run: Any) -> tuple[str, str]:
    branch = run.metadata.get("branch") or git(run).get("branch") or NA
    tags = sorted(str(t) for t in (run.tags or []))
    job_type = getattr(run, "job_type", None) or run.config.get("job_type") or NA
    group = getattr(run, "group", None) or run.config.get("group") or NA
    commit = git(run).get("commit") or getattr(run, "commit", None)

    expected_seed_tag = any(t.lower() == "seed42" for t in tags)
    has_baseline_or_experiment = any(t.lower() in {"baseline", "experiment"} for t in tags)
    has_branch_tag = branch != NA and branch in tags

    lines = [
        f"Entity/project: {run.entity}/{run.project}",
        f"Run name: {run.name}",
        f"Group: {group}",
        f"Job type: {job_type}",
        f"Tags: {', '.join(tags) if tags else NA}",
        f"Branch from metadata: {branch}",
        f"Git commit captured: {'yes' if commit else 'no'}",
        "",
        "Checklist tag checks:",
        f"  seed42 tag: {'yes' if expected_seed_tag else NA}",
        f"  baseline/experiment tag: {'yes' if has_baseline_or_experiment else NA}",
        f"  branch tag matches metadata: {'yes' if has_branch_tag else NA}",
        f"  one-variable tag: {NA}: verify manually from notes.md",
    ]
    emit("wandb_identity.txt", "\n".join(lines))

    complete = (
        run.entity == DEFAULT_ENTITY and run.project == DEFAULT_PROJECT and
        group != NA and job_type != NA and bool(tags) and bool(commit) and
        expected_seed_tag and has_baseline_or_experiment and has_branch_tag
    )
    return (POPULATED if complete else PARTIAL), "W&B identity fields and tag checks"


def show_start_time(run: Any) -> tuple[str, str]:
    ts = iso(run.created_at)
    emit("start_time.txt", ts)
    return POPULATED, ts


def show_end_time(run: Any) -> tuple[str, str]:
    t = end_time(run)
    if t:
        emit("end_time.txt", t)
        return POPULATED, f"created_at + _runtime ({run.summary.get('_runtime', '?'):.0f}s)"
    content = f"{NA}: _runtime missing from summary — fill in manually (UTC ISO)."
    emit("end_time.txt", content)
    return CANNOT, "_runtime not in summary"


def show_status(run: Any) -> tuple[str, str]:
    label = state_label(run)
    emit("status.txt", label)
    return POPULATED, f"{run.state} → {label}"


def show_git_commit(run: Any) -> tuple[str, str]:
    commit = git(run).get("commit") or getattr(run, "commit", None)
    if commit:
        emit("git_commit.txt", commit)
        return POPULATED, commit[:12]
    emit("git_commit.txt",
         f"{NA}: W&B did not capture a git commit.\n"
         "Fill in manually: git rev-parse HEAD")
    return CANNOT, "not captured by W&B"


def show_git_status_before(run: Any) -> tuple[str, str]:
    g = git(run)
    commit = g.get("commit", "")
    branch = run.metadata.get("branch") or g.get("branch", "unknown")
    status = g.get("status", "")

    if status or commit:
        raw = status if isinstance(status, str) else "\n".join(status) if status else "(clean)"
        lines = [
            f"Branch: {branch}",
            "",
            "Git status at launch (reconstructed from W&B metadata):",
            raw,
        ]
        emit("git_status_before.txt", "\n".join(lines))
        return PARTIAL, "reconstructed from run.metadata['git'] — may differ from full git status"

    emit("git_status_before.txt",
         f"{NA}: W&B did not capture git status.\n"
         "Fill in manually from the training VM: git status")
    return CANNOT, "not captured by W&B"


def show_git_status_after(run: Any) -> tuple[str, str]:
    emit("git_status_after.txt",
         f"{NA}: W&B captures git state only at launch, not after training.\n"
         "Fill in from the training VM, or note it was not captured\n"
         "(see baseline_seed42 for precedent).")
    return CANNOT, "W&B does not record post-run git state"


def show_command(run: Any) -> tuple[str, str]:
    program = run.metadata.get("program") or run.metadata.get("codePath") or ""
    args_list = run.metadata.get("args") or []
    python_cmd = f"python -u {program} {' '.join(str(a) for a in args_list)}".strip()

    lines = [
        f"{NA}: Full launcher command (tmux wrapper, env setup) not captured by W&B.",
        "Fill in manually.",
        "",
        "Training entry point (from W&B metadata):",
        python_cmd or f"{NA}: program not recorded in metadata",
    ]
    emit("command.txt", "\n".join(lines))
    return PARTIAL, f"python invocation: {python_cmd!r}" if python_cmd else "no program in metadata"


def show_config_snapshot(run: Any) -> tuple[str, str]:
    # Preferred: download the raw config.py uploaded via wandb.save()
    try:
        for f in run.files():
            if f.name == "config.py":
                with tempfile.TemporaryDirectory() as tmp:
                    f.download(root=tmp, replace=True)
                    src = Path(tmp) / "config.py"
                    if src.exists():
                        emit("config_snapshot.py", src.read_text(encoding="utf-8"))
                        return POPULATED, "raw config.py downloaded from W&B files"
    except Exception:
        pass

    # Fallback: run.config dict (populated for runs using updated train.py)
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    if config:
        lines = [
            "# Config reconstructed from run.config (W&B structured dict).",
            f"# {NA}: imports, derived expressions, and comments are absent.",
            "# Cross-check against the commit in git_commit.txt.",
            "",
        ] + [f"{k} = {v!r}" for k, v in sorted(config.items())]
        emit("config_snapshot.py", "\n".join(lines))
        return PARTIAL, f"{len(config)} values from run.config — missing imports and derived expressions"

    emit("config_snapshot.py",
         f"# {NA}: config.py not in W&B files and run.config is empty.\n"
         "# Copy config.py from the git commit in git_commit.txt manually.")
    return CANNOT, "no config in W&B files or run.config"


def show_completion_summary(run: Any) -> tuple[str, str]:
    commit = git(run).get("commit") or getattr(run, "commit", None) or f"{NA}: see git_commit.txt"
    ckpt_dir = run.config.get("CKPT_DIR") or f"{NA}: see checkpoint_dir.txt"

    lines = [
        f"Run: {run.name}",
        f"Commit: {commit}",
        f"Start: {iso(run.created_at)}",
        f"End: {end_time(run) or f'{NA}: see end_time.txt'}",
        f"Wall-clock: {wall_clock(run)}",
        f"Final step: {final_step(run)}",
        f"Status: {state_label(run)}",
        f"Exit code: {NA}: not recorded by W&B — fill in manually",
        f"Final checkpoint: {ckpt_dir}",
        f"W&B: {run.url}",
        f"Final log marker: {NA}: from raw training log — fill in manually",
    ]
    emit("completion_summary.txt", "\n".join(lines))
    return PARTIAL, "missing exit code, preserved checkpoint copy path, log markers"


def show_checkpoint_dir(run: Any) -> tuple[str, str]:
    ckpt_dir  = run.config.get("CKPT_DIR")              or f"{NA}: fill from CKPT_DIR in config_snapshot.py"
    inter_dir = run.config.get("INTERMEDIATE_CKPT_DIR") or f"{NA}: fill from INTERMEDIATE_CKPT_DIR"
    interval  = run.config.get("SAVE_INTERVAL_STEPS")   or NA
    max_keep  = run.config.get("MAX_TO_KEEP")            or NA

    artifacts = logged_model_artifacts(run)
    artifact_lines: list[str] = []
    artifact_warnings: list[str] = []
    for a in artifacts:
        metadata = dict(getattr(a, "metadata", {}) or {})
        aliases = sorted(str(alias) for alias in getattr(a, "aliases", []) or [])
        has_commit = metadata_has_any(metadata, ("commit", "git_commit", "git_sha", "sha"))
        has_seed = metadata_has_any(metadata, ("seed",))
        artifact_lines.extend([
            f"  {artifact_ref(a)}",
            f"    url: {artifact_url(a)}",
            f"    aliases: {', '.join(aliases) if aliases else NA}",
            f"    metadata has commit: {'yes' if has_commit else NA}",
            f"    metadata has seed: {'yes' if has_seed else NA}",
        ])
        if not has_commit or not has_seed:
            artifact_warnings.append(f"{artifact_ref(a)} missing commit and/or seed metadata")

    lines = [
        "Checkpoint directory (on training VM):",
        str(ckpt_dir),
        "",
        "Intermediate checkpoint directory:",
        str(inter_dir),
        "",
        f"Save interval: {interval} steps",
        f"Max to keep: {max_keep}",
        "",
        f"Preserved local copy: {NA}: filesystem path on training VM — fill in manually",
        "",
        "W&B model artifacts:",
        *(artifact_lines or [f"  {NA}: no model artifact found — upload per CHECKLIST.md §3"]),
        "",
        "Sweep checkpoint coverage:",
        f"  {NA}: verify final plus eval-sweep steps are uploaded as artifact versions.",
    ]
    emit("checkpoint_dir.txt", "\n".join(lines))
    status = PARTIAL if (artifacts or run.config.get("CKPT_DIR")) else CANNOT
    note = f"{len(artifacts)} model artifact(s)"
    if artifact_warnings:
        note += f"; {len(artifact_warnings)} metadata warning(s)"
    return status, note if artifacts else "no model artifact logged"


def show_tensorboard(run: Any) -> tuple[str, str]:
    tb_dir = run.config.get("TENSORBOARD_DIR")
    if tb_dir:
        lines = [
            "TensorBoard log directory:",
            str(tb_dir),
            "",
            f"Configured flush interval: {NA}: not in W&B — check train.py MetricsLoggerOptions",
        ]
        emit("tensorboard.txt", "\n".join(lines))
        return PARTIAL, f"directory from run.config: {tb_dir}"
    emit("tensorboard.txt",
         f"{NA}: TENSORBOARD_DIR not in run.config.\n"
         "Fill from config.py TENSORBOARD_DIR, or note 'not used'.")
    return CANNOT, "not in run.config"


def show_eval_summary(run: Any) -> tuple[str, str]:
    # evaluate.py does not log to W&B; check summary in case it was added manually.
    def find(*keys: str) -> Any:
        for k in keys:
            if k in run.summary:
                return run.summary[k]
        return None

    bac = find("eval/base_acc",     "base_acc")
    bpa = find("eval/base_partial", "base_partial")
    bfo = find("eval/base_format",  "base_format")
    bco = find("eval/base_correct", "base_correct")
    lac = find("eval/lora_acc",     "lora_acc")
    lpa = find("eval/lora_partial", "lora_partial")
    lfo = find("eval/lora_format",  "lora_format")
    lco = find("eval/lora_correct", "lora_correct")

    if not any(v is not None for v in [bac, bpa, bfo, bco, lac, lpa, lfo, lco]):
        emit("eval_summary.txt",
             f"{NA}: evaluate.py does not log to W&B.\n"
             "Add metrics manually via wandb.run.summary.update() per CHECKLIST.md §3,\n"
             "then re-run this script to pick them up.\n"
             "Expected keys: eval/base_acc, eval/base_partial, eval/base_format, eval/base_correct,\n"
             "               eval/lora_acc, eval/lora_partial, eval/lora_format, eval/lora_correct")
        return CANNOT, "evaluate.py does not log to W&B"

    def fmt(v: Any, name: str) -> str:
        if v is None:
            return f"{NA}: {name} not in summary"
        return f"{v:.2f}%" if isinstance(v, float) else str(v)

    lines = [
        "Preset: greedy",
        "Split/sample: GSM8K test, NUM_TEST_BATCHES=64, TRAIN_MICRO_BATCH_SIZE=1",
        "",
        "Base model / zero-init LoRA wrapper:",
        f"  correct={bco or NA}  acc={fmt(bac,'base_acc')}  partial={fmt(bpa,'base_partial')}  format={fmt(bfo,'base_format')}",
        "",
        "Restored LoRA checkpoint:",
        f"  correct={lco or NA}  acc={fmt(lac,'lora_acc')}  partial={fmt(lpa,'lora_partial')}  format={fmt(lfo,'lora_format')}",
    ]
    emit("eval_summary.txt", "\n".join(lines))
    all_found = all(v is not None for v in [bac, bpa, bfo, bco, lac, lpa, lfo, lco])
    return (POPULATED if all_found else PARTIAL), \
           ("all 8 eval metrics found" if all_found else "some eval metrics missing from summary")


def show_eval_logs(run: Any) -> dict[str, tuple[str, str]]:
    """Try to print eval logs from W&B files or artifacts."""
    required = {"base_eval.log", "lora_eval.log"}
    patterns = ("base_eval*.log", "lora_eval*.log", "*_eval_step*.log")
    found: set[str] = set()

    def wanted(name: str) -> bool:
        return any(fnmatch.fnmatch(Path(name).name, pattern) for pattern in patterns)

    try:
        for f in run.files():
            name = Path(f.name).name
            if wanted(name):
                with tempfile.TemporaryDirectory() as tmp:
                    f.download(root=tmp, replace=True)
                    src = Path(tmp) / f.name
                    if src.exists():
                        emit(name, src.read_text(encoding="utf-8", errors="replace"))
                        found.add(name)
    except Exception:
        pass

    try:
        for artifact in run.logged_artifacts():
            for entry_name in artifact.manifest.entries:
                name = Path(entry_name).name
                if wanted(name) and name not in found:
                    with tempfile.TemporaryDirectory() as tmp:
                        artifact.get_path(entry_name).download(root=tmp)
                        src = Path(tmp) / entry_name
                        if src.exists():
                            emit(name, src.read_text(encoding="utf-8", errors="replace"))
                            found.add(name)
    except Exception:
        pass

    results: dict[str, tuple[str, str]] = {}
    for name in sorted(required):
        if name in found:
            results[name] = (POPULATED, "downloaded from W&B")
        else:
            emit(name,
                 f"{NA}: {name} not found in W&B files or artifacts.\n"
                 "Copy from the training VM, or upload as an artifact per CHECKLIST.md §3.")
            results[name] = (CANNOT, "not in W&B files or artifacts")

    sweep_logs = sorted(found - required)
    if sweep_logs:
        results["checkpoint_sweep_eval_logs"] = (POPULATED, ", ".join(sweep_logs))
    else:
        emit("checkpoint_sweep_eval_logs",
             f"{NA}: no checkpoint-sweep eval logs found in W&B files or artifacts.\n"
             "Expected names like lora_eval_step3000.log when sweep evals were run.")
        results["checkpoint_sweep_eval_logs"] = (PARTIAL, "no sweep logs found; ok only if no intermediate checkpoints were evaluated")

    return results


def show_training_curves(run: Any) -> tuple[str, str]:
    summary_keys = {str(k) for k in run.summary.keys()}
    found: dict[str, set[str]] = {"reward": set(), "kl": set(), "loss": set()}

    for key in summary_keys:
        lower = key.lower()
        for family in found:
            if family in lower:
                found[family].add(key)

    missing = [family for family, keys in found.items() if not keys]
    if missing:
        try:
            history = run.history(samples=25)
            for key in history.columns:
                lower = str(key).lower()
                for family in found:
                    if family in lower:
                        found[family].add(str(key))
            missing = [family for family, keys in found.items() if not keys]
        except Exception:
            pass

    lines = [
        "Training curve key checks:",
        f"  reward: {', '.join(sorted(found['reward'])) if found['reward'] else NA}",
        f"  KL: {', '.join(sorted(found['kl'])) if found['kl'] else NA}",
        f"  loss: {', '.join(sorted(found['loss'])) if found['loss'] else NA}",
    ]
    emit("training_curves.txt", "\n".join(lines))
    return (POPULATED if not missing else PARTIAL), \
           ("reward/KL/loss keys found" if not missing else f"missing likely curve keys: {', '.join(missing)}")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(results: dict[str, tuple[str, str]]) -> None:
    by: dict[str, list[tuple[str, str]]] = {POPULATED: [], PARTIAL: [], CANNOT: []}
    for fname, (status, note) in results.items():
        by[status].append((fname, note))

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)

    labels = {POPULATED: "POPULATED", PARTIAL: "PARTIAL (CANNOT_RETRIEVE markers above)", CANNOT: "CANNOT RETRIEVE"}
    for status in (POPULATED, PARTIAL, CANNOT):
        if by[status]:
            print(f"\n  {labels[status]} ({len(by[status])})")
            for f, n in by[status]:
                print(f"    {f:<35} {n}")

    manual = [
        "notes.md: changed vs baseline, why, what happened, next action",
        "failure_summary.txt: required for failed runs",
        "docs/RUNS.md: planned/final row, W&B log link, eval acc, status, notes",
        "branch/commit/PR process items",
        "one-variable-change claim and constants held fixed",
    ]
    print("\n  MANUAL / NOT COVERED BY W&B")
    for item in manual:
        print(f"    {item}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    run = fetch_run(args)

    results: dict[str, tuple[str, str]] = {}
    results["wandb_url.txt"]          = show_wandb_url(run)
    results["wandb_identity.txt"]     = show_wandb_identity(run)
    results["start_time.txt"]         = show_start_time(run)
    results["end_time.txt"]           = show_end_time(run)
    results["status.txt"]             = show_status(run)
    results["git_commit.txt"]         = show_git_commit(run)
    results["git_status_before.txt"]  = show_git_status_before(run)
    results["git_status_after.txt"]   = show_git_status_after(run)
    results["command.txt"]            = show_command(run)
    results["config_snapshot.py"]     = show_config_snapshot(run)
    results["completion_summary.txt"] = show_completion_summary(run)
    results["checkpoint_dir.txt"]     = show_checkpoint_dir(run)
    results["tensorboard.txt"]        = show_tensorboard(run)
    results["eval_summary.txt"]       = show_eval_summary(run)
    results["training_curves.txt"]    = show_training_curves(run)
    results.update(show_eval_logs(run))

    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
