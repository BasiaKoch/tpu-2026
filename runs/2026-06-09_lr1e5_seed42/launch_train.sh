#!/usr/bin/env bash
set -o pipefail

cd /home/funmilooi-somoye/tpu-2026/scripts || exit 1
source ~/venvs/tunix/bin/activate

set -a
[ -f ~/.env ] && source ~/.env
set +a

WANDB_ENTITY=felsomoye-university-of-cambridge \
WANDB_PROJECT=tunix \
python -u train.py 2>&1 | tee /home/funmilooi-somoye/tpu-2026/runs/2026-06-09_lr1e5_seed42/live_log.txt

code=${PIPESTATUS[0]}
printf "\n--- train.py exited with code %s ---\n" "$code"
printf "%s\n" "$code" > /home/funmilooi-somoye/tpu-2026/runs/2026-06-09_lr1e5_seed42/exit_code.txt
exec bash
