#!/usr/bin/env bash
set -uo pipefail

RUN_DIR="/home/funmilooi-somoye/tpu-2026/runs/baseline_seed42_to5864"
SCRIPT_DIR="/home/funmilooi-somoye/tpu-2026/scripts"
VENV="/home/funmilooi-somoye/venvs/tunix/bin/activate"
CKPT_DIR="/home/funmilooi-somoye/tpu-2026/checkpoints/baseline_seed42_to5864/actor"
FULL_TEST_COUNT=1319

mkdir -p "$RUN_DIR"
cd "$SCRIPT_DIR"
source "$VENV"
if [ -f ~/.env ]; then
  set -a
  source ~/.env
  set +a
fi

{
  echo "Status: running"
  echo
  echo "Tmux:"
  echo "  session: tunix_resume_5864"
  echo "  window: full_eval_5864"
  echo
  echo "Launch:"
  echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "Full test examples:"
  echo "  ${FULL_TEST_COUNT}"
  echo
  echo "Logs:"
  echo "  Base: ${RUN_DIR}/base_eval_full.log"
  echo "  LoRA step 5864: ${RUN_DIR}/lora_eval_step5864_full.log"
} > "${RUN_DIR}/eval_full_status.txt"

run_eval() {
  local label="$1"
  local log_path="$2"
  shift 2

  echo "Running ${label}: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_DIR}/eval_full_status.txt"
  python -u - "$@" <<'PY' 2>&1 | tee "$log_path"
import sys
import evaluate

sys.argv = sys.argv[1:]
evaluate.NUM_TEST_BATCHES = 1319
evaluate.main()
PY
  local status=${PIPESTATUS[0]}
  echo "${label} exit: ${status} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_DIR}/eval_full_status.txt"
  return "$status"
}

run_eval "base full eval" "${RUN_DIR}/base_eval_full.log" \
  evaluate.py --preset greedy --source tfds --no-restore
base_status=$?

run_eval "LoRA step 5864 full eval" "${RUN_DIR}/lora_eval_step5864_full.log" \
  evaluate.py --preset greedy --source tfds --ckpt-dir "$CKPT_DIR" --step 5864
lora_status=$?

base_final=$(grep 'FINAL:' "${RUN_DIR}/base_eval_full.log" | tail -n 1 || true)
lora_final=$(grep 'FINAL:' "${RUN_DIR}/lora_eval_step5864_full.log" | tail -n 1 || true)
end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
  echo "Baseline seed 42 continuation full-test eval summary"
  echo "======================================================"
  echo
  echo "Preset:"
  echo "  greedy"
  echo
  echo "Source:"
  echo "  tfds"
  echo
  echo "Split/sample:"
  echo "  full GSM8K test split, ${FULL_TEST_COUNT} examples, TRAIN_MICRO_BATCH_SIZE=1"
  echo
  echo "Base model / zero-init LoRA wrapper:"
  echo "  Log: ${RUN_DIR}/base_eval_full.log"
  echo "  ${base_final}"
  echo
  echo "Restored LoRA checkpoint:"
  echo "  Checkpoint root: ${CKPT_DIR}"
  echo "  Step: 5864"
  echo "  Log: ${RUN_DIR}/lora_eval_step5864_full.log"
  echo "  ${lora_final}"
  echo
  echo "Exit status:"
  echo "  Base: ${base_status}"
  echo "  LoRA step 5864: ${lora_status}"
  echo
  echo "End:"
  echo "  ${end_time}"
} > "${RUN_DIR}/eval_full_summary.txt"

{
  echo
  echo "Results:"
  echo "  Base: ${base_final}"
  echo "  LoRA step 5864: ${lora_final}"
  echo
  echo "Exit status:"
  echo "  Base: ${base_status}"
  echo "  LoRA step 5864: ${lora_status}"
  echo
  echo "End:"
  echo "  ${end_time}"
  if [ "$base_status" -eq 0 ] && [ "$lora_status" -eq 0 ]; then
    echo
    echo "Status: completed"
  else
    echo
    echo "Status: failed"
  fi
} >> "${RUN_DIR}/eval_full_status.txt"

echo
echo "--- full eval process exited ---"
