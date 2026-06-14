#!/usr/bin/env bash
#
# run_bootstrap.sh — bootstrap 95% CIs for GSM8K accuracy (see docs/BOOTSTRAP.md).
#
# Interactive: prompts for which run you are bootstrapping (a short label) and
# the W&B checkpoint artifact URL for that run's fine-tuned LoRA model. It then:
#   1. evaluates the BASE gemma-3-1b-it (no fine-tuning) over the full 1319-question
#      GSM8K test split   -> analysis/base_no_ft.jsonl
#   2. downloads the W&B checkpoint and evaluates the fine-tuned LoRA model
#      over the same split -> analysis/<label>_lora.jsonl
#   3. runs the percentile bootstrap (10k iters, seed 42) on both
#   4. aggregates both into analysis/bootstrap_results_<label>.txt
#
# Idempotent: the expensive eval step is SKIPPED if the .jsonl already exists, so
# re-runs (or a machine without a TPU) just redo the cheap bootstrap. Delete the
# .jsonl to force a fresh eval. Set FORCE_EVAL=1 to regenerate unconditionally.
#
# Env overrides: N_ITER (default 10000), SEED (default 42), VENV, NUM_TEST_BATCHES,
# CHECKPOINT_PATH (restore from a local checkpoint dir instead of a W&B artifact).
#
# Usage (from the repo root):
#   ./scripts/run_bootstrap.sh
#   ./scripts/run_bootstrap.sh k8 https://wandb.ai/<entity>/<project>/artifacts/model/<name>
#   CHECKPOINT_PATH=~/checkpoints/k-8-new-reward/actor/5864 ./scripts/run_bootstrap.sh k8
#   (positional args skip the prompts; either or both may be omitted. A local
#    CHECKPOINT_PATH takes precedence over the W&B URL.)

set -euo pipefail

# This script lives in scripts/; the repo root is its parent directory.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$HOME/venvs/tunix}
ANALYSIS="$REPO/analysis"
N_ITER=${N_ITER:-10000}
SEED=${SEED:-42}
# Number of test questions to eval (1 question per batch at TRAIN_MICRO_BATCH_SIZE=1).
# Default = the full 1319-question GSM8K test split.
NUM_TEST_BATCHES=${NUM_TEST_BATCHES:-1319}
# Restore from a local checkpoint dir (e.g. .../actor/5864) instead of a W&B
# artifact. When set, the W&B URL is not required.
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}

RUN_LABEL=${1:-}
WANDB_URL=${2:-}

if [[ -z "$RUN_LABEL" ]]; then
  read -r -p "Which run are you bootstrapping for? (short label, e.g. k8): " RUN_LABEL
fi
RUN_LABEL=$(echo "$RUN_LABEL" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_//;s/_$//')
if [[ -z "$RUN_LABEL" ]]; then
  echo "ERROR: a run label is required." >&2
  exit 1
fi

# A local checkpoint path takes precedence over the W&B URL. Only prompt for /
# require the URL when no local path was provided.
if [[ -n "$CHECKPOINT_PATH" ]]; then
  if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: CHECKPOINT_PATH not found: $CHECKPOINT_PATH" >&2
    exit 1
  fi
else
  if [[ -z "$WANDB_URL" ]]; then
    read -r -p "W&B checkpoint artifact URL for the fine-tuned LoRA model: " WANDB_URL
  fi
  if [[ -z "$WANDB_URL" ]]; then
    echo "ERROR: a W&B checkpoint URL is required (or set CHECKPOINT_PATH)." >&2
    exit 1
  fi
fi

mkdir -p "$ANALYSIS"

BASE_JSONL="$ANALYSIS/base_no_ft.jsonl"
LORA_JSONL="$ANALYSIS/${RUN_LABEL}_lora.jsonl"
RESULTS="$ANALYSIS/bootstrap_results_${RUN_LABEL}.txt"

# --- helpers -------------------------------------------------------------------

# Run `bootstrap.py eval` (needs the TPU env: venv + ~/.env secrets).
run_eval() {
  if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV — needed to generate the .jsonl." >&2
    echo "       Set up the environment (see bootstrap.sh / tpu-setup.md) or" >&2
    echo "       provide the .jsonl files in analysis/ and re-run." >&2
    exit 1
  fi
  ( source "$VENV/bin/activate" \
    && { [ -f "$HOME/.env" ] && set -a && source "$HOME/.env" && set +a || true; } \
    && python -u "$REPO/scripts/bootstrap.py" eval --num-test-batches "$NUM_TEST_BATCHES" "$@" )
}

needs_eval() {
  [[ "${FORCE_EVAL:-0}" == "1" || ! -s "$1" ]]
}

# --- 1. base model -------------------------------------------------------------

if needs_eval "$BASE_JSONL"; then
  echo "==> Evaluating BASE gemma-3-1b-it (no fine-tuning) over $NUM_TEST_BATCHES questions"
  run_eval --no-restore --dump-jsonl "$BASE_JSONL"
else
  echo "==> Reusing existing $BASE_JSONL (skip eval; set FORCE_EVAL=1 to redo)"
fi

# --- 2. fine-tuned LoRA --------------------------------------------------------

if needs_eval "$LORA_JSONL"; then
  if [[ -n "$CHECKPOINT_PATH" ]]; then
    echo "==> Evaluating fine-tuned LoRA ($RUN_LABEL) from local checkpoint $CHECKPOINT_PATH"
    run_eval --checkpoint-path "$CHECKPOINT_PATH" --dump-jsonl "$LORA_JSONL"
  else
    echo "==> Downloading checkpoint + evaluating fine-tuned LoRA ($RUN_LABEL)"
    run_eval --wandb-artifact "$WANDB_URL" --dump-jsonl "$LORA_JSONL"
  fi
else
  echo "==> Reusing existing $LORA_JSONL (skip eval; set FORCE_EVAL=1 to redo)"
fi

# --- 3 + 4. bootstrap both and aggregate --------------------------------------

# Prefer the venv python (has numpy); fall back to whatever python3 is on PATH.
PY=python3
if [[ -d "$VENV" ]]; then PY="$VENV/bin/python"; fi

echo "==> Bootstrapping ($N_ITER iters, seed $SEED) -> $RESULTS"
"$PY" "$REPO/scripts/bootstrap.py" ci "$BASE_JSONL" \
  --label "base gemma-3-1b-it (no fine-tuning)" \
  --n-iter "$N_ITER" --seed "$SEED" --output "$RESULTS"
"$PY" "$REPO/scripts/bootstrap.py" ci "$LORA_JSONL" \
  --label "fine-tuned LoRA — run ${RUN_LABEL}" \
  --n-iter "$N_ITER" --seed "$SEED" --output "$RESULTS" --append

echo
echo "==> Done. Aggregated results:"
cat "$RESULTS"
