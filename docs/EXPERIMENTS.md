# Experiments

Pre-register each experiment before launching a full TPU run.

## Experiment Summary

| Experiment | Main change | Hypothesis | Success metric | Diagnostics | Status |
|---|---|---|---|---|---|
| Baseline GRPO | Default `scripts/config.py` and rewards (`NUM_GENERATIONS = 2`). | Baseline should reproduce the expected reward/KL behaviour and improve over the base model. | GSM8K eval accuracy. | Reward curve, KL curve, output format accuracy. | completed (run `jgs4c6kl`); LoRA eval below base on 64-example greedy eval |
| Group size G=8 | `NUM_GENERATIONS` 2 → 8 in `scripts/config.py`. | A larger GRPO group gives a less noisy advantage baseline/normalisation, so policy updates are more stable and avoid the late-training collapse seen in the baseline. | Eval reward mean (and GSM8K accuracy) vs. baseline. | Eval/train reward curves, KL, grad-norm spikes, completion length, empty-completion rate. | running |
| Micro batch size 2 | `TRAIN_MICRO_BATCH_SIZE` 1 → 2 in `scripts/config.py`. | A larger micro-batch should improve TPU compute efficiency by doing more useful work per step, as long as memory headroom is sufficient and training dynamics remain comparable. | Wall-clock time per step and eval reward/GSM8K accuracy vs. baseline. | TPU utilisation, step time, OOMs, KL, grad norm, reward curves. | planned |
| Variant A: reward/length | Add a mild length penalty or adjust shaping reward. | Reducing verbosity or reward hacking may improve correctness without excessive KL drift. | GSM8K eval accuracy vs. baseline. | Response length, malformed-output rate, correctness reward. | planned |
| Variant B: KL control | `BETA` 0.08 → 1e-6 (effectively off) and 0.08 → 0.32 (two runs; env-var override, K=2 base). | β is the only active trust region (clip never binds at μ=1), so β≈0 should collapse harder/faster and β=0.32 should contain KL drift and delay/prevent collapse. | GSM8K eval accuracy read alongside KL, vs. baseline `jgs4c6kl`. | KL vs. step, KL vs. accuracy, reward curve, completion length. | registered (`kl-control`) |

## Logged Experiments

### `kl-control`

**Question:** Is weak KL control a cause of the baseline's late-training collapse? Equivalently: what does the KL-budget axis (β) do to stability and held-out accuracy?

**Main change:** `BETA` only, via a per-run environment variable (`scripts/config.py` now reads `BETA = float(os.environ.get("BETA", "0.08"))`; default unchanged). Two runs:
- `kl_beta1e-6_g2_seed42` — `BETA=1e-6` (reference-KL penalty **effectively off**; see amendment below for why not literal 0).
- `kl_beta032_g2_seed42` — `BETA=0.32` (4× the baseline leash).

**Amendment (pre-launch):** the original registration used `BETA=0.0`, but tunix's `GRPOLearner` guards the reference-model pass with `if self.algo_config.beta != 0.0:` — at literal 0 it skips reference log-probs entirely and **never computes or logs the KL metric**, deleting this run's primary diagnostic and skipping compute the control performs (confounding wall-clock). `BETA=1e-6` makes the penalty numerically negligible (≈4×10⁻⁵ at the baseline's worst KL≈41, vs. rewards of order 1–10) while keeping the KL curve logged and the per-step compute path identical to the control.

**Hypothesis:** The baseline logs show `actor/train/pg_clipfrac = 0` throughout: with `NUM_ITERATIONS = 1`, the policy that generates rollouts is the policy being updated, so the importance ratio is 1 and the PPO clip ε never binds. The β·KL(π_θ‖π_ref) penalty is therefore the **only active trust region**. Prediction: β≈0 (1e-6) removes the last effective constraint on policy movement → faster/more severe collapse and unbounded KL drift; β=0.32 contains KL drift and delays or prevents the collapse (the baseline at β=0.08 spiked to KL≈41), possibly at the cost of slower reward growth. Together with the β=0.08 baseline these give a 3-point KL-budget dose–response curve.

**Config:** Branch `kl-control-bk` (off `origin/baseline-fls`); control run `jgs4c6kl` at commit `7e696c4`. Everything held at baseline values: `NUM_GENERATIONS = 2`, `LEARNING_RATE = 3e-6`, `RANK = 64`, `EPSILON = 0.2`, `TRAIN_MICRO_BATCH_SIZE = 1`, `MAX_STEPS = 3364`, same TFDS data and eval split, data shuffle seed 42 (hard-coded in `scripts/data.py`, identical to the control). The only varied quantity is β.

**Comparison:** Both runs against baseline `jgs4c6kl` (G=2, β=0.08) — see [RUNS.md](RUNS.md). Caveat: these runs execute on a different v6e-1 VM than the baseline (same accelerator generation, libraries installed from the same unpinned GitHub HEAD recipe); conclusions are read primarily from trajectory shape (collapse vs. containment, KL drift), which is robust to this.

**Metrics:** Mean reward vs. step; KL vs. step (primary); eval reward vs. step; **KL vs. held-out accuracy** across the three β values (the exam's suggested diagnostic); completion length vs. step; GSM8K accuracy across retained checkpoints.

**Complements:** Rowan's group-size sweep (K = 4/8/16) probes advantage-estimator variance, Funmi's `learning-rate-fls` probes optimiser step size on the G8 base, and the `reward-length` variants probe the reward channel. This experiment covers the remaining named axis — the KL trust region — so the team's variants together span group size, step size, reward shaping, and KL budget.

**Result:** _Pending._

**Interpretation:** _Pending._

## Logged Experiments

### `group-size-g8`

**Question:** Does increasing the GRPO group size `NUM_GENERATIONS` from 2 to 8 fix the unstable, collapsing training seen in the baseline?

**Main change:** `NUM_GENERATIONS = 2 → 8` in [scripts/config.py:44](../scripts/config.py). All other hyperparameters unchanged.

**Hypothesis:** At G=2 the per-prompt advantage baseline is a 2-sample mean and the std-normalisation is extremely noisy, producing high-variance policy gradients. This is the likely driver of the baseline's late-training collapse (KL spike to 41, grad-norm spike to 696, completions degenerating to 0 length — see [baseline-analysis.md](rowan/baseline-analysis.md)). G=8 should give a much lower-variance advantage estimate, keeping KL and grad-norm controlled and sustaining the early reward gains instead of collapsing.

**Config:** Branch TBD; commit TBD (the `2 → 8` edit). Seed: not yet fixed. Data: GSM8K, `TRAIN_FRACTION = 0.9`. Steps: `MAX_STEPS = 3364`. Changed parameter: `NUM_GENERATIONS = 8`.

**Comparison:** Against baseline run `jgs4c6kl` (G=2) — see [RUNS.md](RUNS.md).

**Metrics:** Eval reward mean, `rewards/eval/check_answer`, KL (`actor/train/kl`), grad-norm, completion length, empty-completion rate, format-match rewards.

**Result:** _Pending — fill in after the run completes._

**Interpretation:** _Pending._

Baseline reference (G=2, `jgs4c6kl`): peak eval reward mean **1.711** at step **448**, collapsing to **-0.240** by the final eval (step 3328); wall-clock ≈ **4.7 h** over 3,364 steps.

### `micro-batch-size-2`

**Question:** Does increasing `TRAIN_MICRO_BATCH_SIZE` from 1 to 2 improve TPU compute efficiency without hurting training quality?

**Main change:** `TRAIN_MICRO_BATCH_SIZE = 1 → 2` in [scripts/config.py:52](../scripts/config.py). All other hyperparameters should stay unchanged for the controlled comparison.

**Hypothesis:** With a larger micro-batch, each training step should do more useful work per compile/execution overhead and improve TPU utilisation, reducing wall-clock time per effective amount of training. The main risk is insufficient memory headroom or a change in optimisation behaviour from the larger per-step batch.

**Config:** Branch TBD; commit TBD. Seed: not yet fixed. Data: GSM8K, `TRAIN_FRACTION = 0.9`. Steps: `MAX_STEPS = 3364`. Changed parameter: `TRAIN_MICRO_BATCH_SIZE = 2`.

**Comparison:** Against baseline run `jgs4c6kl` (`TRAIN_MICRO_BATCH_SIZE = 1`) — see [RUNS.md](RUNS.md). Compare both compute metrics and model-quality metrics, not step time alone.

**Metrics:** Wall-clock time per step, total runtime, TPU utilisation if available, eval reward mean, `rewards/eval/check_answer`, GSM8K accuracy, KL (`actor/train/kl`), grad-norm, OOM/retry events.

**Result:** _Pending — fill in after the run completes._

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
