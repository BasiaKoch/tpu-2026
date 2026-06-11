# 2026-06-10_kl_beta1e-6_g2_seed42

Changed vs. baseline: `BETA = 1e-6` (reference-KL penalty effectively off). Everything
else held constant: `NUM_GENERATIONS = 2`, `LEARNING_RATE = 3e-6`,
`TRAIN_MICRO_BATCH_SIZE = 1`, `RANK = 64`, `EPSILON = 0.2`, data shuffle seed 42,
`MAX_STEPS = 3364`, same TFDS data and greedy 64-example eval protocol.

Why 1e-6 and not literal 0: tunix's `GRPOLearner` guards the reference-model pass with
`if beta != 0.0` — at exactly 0 it skips reference log-probs entirely and never computes
or logs the KL metric, which is this run's primary diagnostic, and it also skips compute
the control performs (confounding wall-clock). At 1e-6 the penalty term is numerically
negligible (~4e-5 at the baseline's worst KL of ~41, vs. rewards of order 1-10) while
KL logging and the per-step compute path stay identical to the control. Registered in
`docs/EXPERIMENTS.md` under `kl-control` (amendment noted there).

Hypothesis: the baseline logs show `actor/train/pg_clipfrac = 0` throughout — with
`NUM_ITERATIONS = 1` the rollout policy equals the updated policy, the importance ratio
is identically 1, and the PPO clip never binds. The beta*KL penalty is therefore the only
active trust region. Removing it (beta ~ 0) should produce faster/larger KL drift and an
earlier or more severe version of the baseline's collapse (baseline: KL spike to ~41,
eval reward 1.711 at step ~448 collapsing to -0.240).

Observed: pending — fill in from the W&B curves (`actor/train/kl`, eval reward,
completion length) compared against baseline `jgs4c6kl`, alongside the beta=0.32 partner
run (`kl_beta032_g2_seed42`).

Caveats: run on Barbara's v6e-1 VM, not the VM that produced the baseline (same
accelerator generation; libraries installed via the same unpinned-HEAD bootstrap).
Conclusions are read primarily from trajectory shape, which is robust to this.

Evaluation: pending — greedy 64-example suite (base + restored checkpoint + sweep over
saved steps 2000/2500/3000/3364) after the beta=0.32 run frees the chip.

W&B checkpoint artifact: `felsomoye-university-of-cambridge/tunix/kl_beta1e6_g2_seed42_checkpoint:latest`.
Artifact upload run: https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/p0y6k89y.
