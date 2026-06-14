Changed vs. baseline: trained on the local hard/medium JSONL dataset, and NUM_GENERATIONS=8. Everything else held constant unless noted in config_snapshot.py.

Failure update, 2026-06-13:
- W&B run hlo1w4go stopped because the VM rebooted at about 2026-06-13T13:12:25Z.
- The training process and tmux session disappeared without a Python traceback or clean W&B finish.
- Checkpoints were under /tmp/content/ckpts/actor/ and were lost when /tmp/content disappeared after reboot.
- Relaunch from scratch with a persistent checkpoint root; do not resume hlo1w4go because the weights are gone.
