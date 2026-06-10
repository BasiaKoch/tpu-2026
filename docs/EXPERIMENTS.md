# Experiments

Pre-register each experiment before launching a full TPU run.

## Experiment Summary

| Experiment | Main change | Hypothesis | Success metric | Diagnostics | Status |
|---|---|---|---|---|---|
| Baseline GRPO | Default `scripts/config.py` and rewards. | Baseline should reproduce the expected reward/KL behaviour and improve over the base model. | GSM8K eval accuracy. | Reward curve, KL curve, output format accuracy. | completed; LoRA eval below base on 64-example greedy eval |
| Variant A: reward/length | Add a mild length penalty or adjust shaping reward. | Reducing verbosity or reward hacking may improve correctness without excessive KL drift. | GSM8K eval accuracy vs. baseline. | Response length, malformed-output rate, correctness reward. | planned |
| Variant B: KL control | `BETA` 0.08 → 0.0 and 0.08 → 0.32 (two runs; env-var override, K=2 base). | β is the only active trust region (clip never binds at μ=1), so β=0 should collapse harder/faster and β=0.32 should contain KL drift and delay/prevent collapse. | GSM8K eval accuracy read alongside KL, vs. baseline `jgs4c6kl`. | KL vs. step, KL vs. accuracy, reward curve, completion length. | registered (`kl-control`) |

## Logged Experiments

### `kl-control`

**Question:** Is weak KL control a cause of the baseline's late-training collapse? Equivalently: what does the KL-budget axis (β) do to stability and held-out accuracy?

**Main change:** `BETA` only, via a per-run environment variable (`scripts/config.py` now reads `BETA = float(os.environ.get("BETA", "0.08"))`; default unchanged). Two runs:
- `kl_beta0_g2_seed42` — `BETA=0.0` (reference-KL penalty **off**, as the exam's I.3 "algorithmic" direction names).
- `kl_beta032_g2_seed42` — `BETA=0.32` (4× the baseline leash).

**Hypothesis:** The baseline logs show `actor/train/pg_clipfrac = 0` throughout: with `NUM_ITERATIONS = 1`, the policy that generates rollouts is the policy being updated, so the importance ratio is 1 and the PPO clip ε never binds. The β·KL(π_θ‖π_ref) penalty is therefore the **only active trust region**. Prediction: β=0 removes the last constraint on policy movement → faster/more severe collapse and unbounded KL drift; β=0.32 contains KL drift and delays or prevents the collapse (the baseline at β=0.08 spiked to KL≈41), possibly at the cost of slower reward growth. Together with the β=0.08 baseline these give a 3-point KL-budget dose–response curve.

**Config:** Branch `kl-control-bk` (off `origin/baseline-fls`); control run `jgs4c6kl` at commit `7e696c4`. Everything held at baseline values: `NUM_GENERATIONS = 2`, `LEARNING_RATE = 3e-6`, `RANK = 64`, `EPSILON = 0.2`, `TRAIN_MICRO_BATCH_SIZE = 1`, `MAX_STEPS = 3364`, same TFDS data and eval split, data shuffle seed 42 (hard-coded in `scripts/data.py`, identical to the control). The only varied quantity is β.

**Comparison:** Both runs against baseline `jgs4c6kl` (G=2, β=0.08) — see [RUNS.md](RUNS.md). Caveat: these runs execute on a different v6e-1 VM than the baseline (same accelerator generation, libraries installed from the same unpinned GitHub HEAD recipe); conclusions are read primarily from trajectory shape (collapse vs. containment, KL drift), which is robust to this.

**Metrics:** Mean reward vs. step; KL vs. step (primary); eval reward vs. step; **KL vs. held-out accuracy** across the three β values (the exam's suggested diagnostic); completion length vs. step; GSM8K accuracy across retained checkpoints.

**Complements:** Rowan's group-size sweep (K = 4/8/16) probes advantage-estimator variance, Funmi's `learning-rate-fls` probes optimiser step size on the G8 base, and the `reward-length` variants probe the reward channel. This experiment covers the remaining named axis — the KL trust region — so the team's variants together span group size, step size, reward shaping, and KL budget.

**Result:** _Pending._

**Interpretation:** _Pending._

## Template

### `<experiment-name>`

**Question:** What does this experiment test?

**Main change:** One controlled code/config change.

**Hypothesis:** What should happen, and why?

**Config:** Branch, commit, seed, data split, total steps, changed parameters.

**Comparison:** Which baseline/run is this compared against?

**Metrics:** Accuracy, reward, KL, response length, format accuracy, or other diagnostics.

**Result:** Fill in after evaluation.

**Interpretation:** Did the outcome match the hypothesis?

