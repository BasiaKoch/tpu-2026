# I.1 Baseline Results

Shared factual record for report section I.1. Report prose should still be written independently.

## Run Metadata

| Item | Value |
|---|---|
| Run | `baseline_seed42` |
| Commit | `7e696c428687bc083604690f5f51009da6abb6d9` |
| Branch | `baseline-fls` |
| Config | Default `scripts/config.py`, except environment/runtime patch documented below |
| TPU | v6e-1 |
| Start | `2026-06-08T18:17:09Z` |
| End | `2026-06-08T22:58:47Z` |
| Wall-clock | 4h 41m 38s |
| GRPO steps completed | 3364 |
| Training W&B run | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl |
| Final checkpoint | `/tmp/content/ckpts/actor/3364` |
| Preserved checkpoint | `checkpoints/baseline_seed42/actor/3364` |
| W&B artifact | `felsomoye-university-of-cambridge/tunix/baseline_seed42_checkpoint:latest` |

## Evaluation Results

Evaluation used `scripts/evaluate.py` with greedy decoding on the configured held-out GSM8K test eval: `NUM_TEST_BATCHES=64`, `TRAIN_MICRO_BATCH_SIZE=1`, TFDS source, dataset shuffle seed `42` from `scripts/data.py`.

| Model | Checkpoint | Correct | Accuracy | Partial accuracy | Format accuracy | Log |
|---|---|---:|---:|---:|---:|---|
| Base `gemma-3-1b-it` / no restored LoRA | n/a | 33/64 | 51.56% | 53.12% | 6.25% | `runs/2026-06-08_baseline_seed42/base_eval.log` |
| LoRA-finetuned baseline | step 3364 | 2/64 | 3.12% | 6.25% | 12.50% | `runs/2026-06-08_baseline_seed42/lora_eval.log` |

The final LoRA checkpoint was substantially worse than the base model on this evaluation. The checkpoint restored successfully, and retained-checkpoint evals show progressive degradation rather than a one-off bad final copy:

| Checkpoint | Accuracy |
|---|---:|
| Base / no restore | 51.56% |
| step 2000 | 28.12% |
| step 2500 | 20.31% |
| step 3000 | 6.25% |
| step 3364 | 3.12% |

Detailed diagnostic note: `runs/2026-06-08_baseline_seed42/degradation_diagnostic.txt`.

## Training Curves

Generated from the TensorBoard event file
`/tmp/content/tmp/tensorboard/grpo/events.out.tfevents.1780942641.t1v-n-f769470f-w-0`.
The plotting script and source curve assets are retained on the pushed
[`analysis-bk`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/tree/analysis-bk)
branch.

| Asset | File |
|---|---|
| Plotting script | [`analysis/plot_baseline_curves.py`](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/blob/analysis-bk/analysis/plot_baseline_curves.py) |
| Mean reward curve | [PNG](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/blob/analysis-bk/report_assets/baseline_mean_reward_curve.png) · [PDF](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/blob/analysis-bk/report_assets/baseline_mean_reward_curve.pdf) · [CSV](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/blob/analysis-bk/report_assets/baseline_mean_reward_curve.csv) |
| KL curve | [PNG](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/blob/analysis-bk/report_assets/baseline_kl_curve.png) · [PDF](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/blob/analysis-bk/report_assets/baseline_kl_curve.pdf) · [CSV](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/rd761/-/blob/analysis-bk/report_assets/baseline_kl_curve.csv) |

## Baseline Patch

The baseline did not run as-shipped initially because TFDS/GSM8K loading failed with a protobuf `FieldDescriptor.label` error. The only accepted baseline patch was pinning `protobuf==6.31.1` for compatibility with `tensorflow-datasets==4.9.9`. This is environment/dependency plumbing only; it did not change the model, reward functions, data split, or training hyperparameters.

Patch details are in `docs/BASELINE_PATCHES.md` and setup history is in `docs/SETUP_NOTES.md`.
