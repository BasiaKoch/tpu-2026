#!/usr/bin/env bash
# Launch training inside a detached tmux session so closing your shell does
# NOT kill the run. Re-run this script and it just attaches to the session.
#
#   ./run_tmux.sh                # start (or attach)
#   ./run_tmux.sh resume         # resume the wandb run id below
#   tmux attach -t tunix         # reattach manually
#   tmux kill-session -t tunix   # stop everything

set -euo pipefail

SESSION=tunix
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$HOME/venvs/tunix}
WANDB_RUN_ID="${WANDB_RUN_ID:-bnh9ttlt}"   # the run that was interrupted
TPU_CONTENT_DIR="${TPU_CONTENT_DIR:-$REPO/checkpoints/hard_medium_jsonl_g8_seed42_rerun_content}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists — attaching."
  exec tmux attach -t "$SESSION"
fi

INNER="cd $REPO/scripts && mkdir -p $TPU_CONTENT_DIR && source $VENV/bin/activate && [ -f ~/.env ] && set -a && source ~/.env && set +a && TPU_CONTENT_DIR=$TPU_CONTENT_DIR python -u train.py"
if [[ "${1:-}" == "resume" ]]; then
  INNER="cd $REPO/scripts && mkdir -p $TPU_CONTENT_DIR && source $VENV/bin/activate && [ -f ~/.env ] && set -a && source ~/.env && set +a && TPU_CONTENT_DIR=$TPU_CONTENT_DIR WANDB_RUN_ID=$WANDB_RUN_ID python -u train.py --wandb-run-id $WANDB_RUN_ID"
fi

# Run under bash (not dash) so `source` works, and keep the shell alive on
# success/failure so we can read the output instead of tmux closing on us.
# Also tee output to a logfile in case we miss something on screen.
LOG="$REPO/scripts/train.log"
CMD="bash -lc '$INNER 2>&1 | tee -a $LOG; echo; echo \"--- process exited (\$?) ---\"; exec bash'"

tmux new-session -d -s "$SESSION" "$CMD"
echo "Started tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
echo "Log file:                                tail -f $LOG"
echo "Checkpoint root:                         $TPU_CONTENT_DIR"
