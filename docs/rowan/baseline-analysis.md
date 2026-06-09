Analysed `wandb_jgs4c6kl_history.csv`

**Headline:** this run learned something early, then over-trained or destabilized badly. The best eval point is around **step 448**, not the end.

Key findings:

- Run span: **3,365 rows**, steps **0-3364**, logged from **2026-06-08 18:23:21 UTC** to **22:58:44 UTC**.
- Eval logged every ~64 steps, with **53 eval points**.
- Best eval reward:
  - **step 448**
  - `rewards/eval/mean = 1.711`
  - `rewards/eval/score/mean = 6.845`
- Final eval at step 3328:
  - `rewards/eval/mean = -0.240`
  - `rewards/eval/score/mean = -0.960`
- Worst eval was near **step 2880**, with `rewards/eval/mean = -0.259`.

The collapse is mostly formatting-related. `rewards/eval/match_format_approximately` peaks at **2.337** around step 704, then falls to about **-1.57** by the end. Exact format reward also falls from a peak of **2.755** to **0.389**. Answer correctness also weakens, but less dramatically.

Completion lengths are suspicious: eval mean length rises as high as **471.6** around step 1728, then collapses to about **118-136** late in training. Train rows near the end include several **0-length completions**, which strongly suggests the model/generation path is degenerating, possibly emitting EOS immediately or failing the response template.

Training reward tells the same story: the best 50-step rolling window ends at **step 479** with mean reward **1.866**; the worst ends at **step 3155** with mean reward **-0.508**. The final decile of training has mean train reward **-0.256**.

Diagnostics worth noting:

- `actor/train/pg_clipfrac` is **always 0**, so clipping either never activates or the metric is not meaningful as logged.
- Train KL rises over time and has a huge spike: max **41.0** at step 3293.
- Grad norm also spikes: max **696** at step 2187.
- CSV only records one checkpoint/write metric row at step 0, so the history file alone does not confirm later checkpoint saves.

Recommendation: for reporting or adapter selection, treat **step 448**, or roughly the **step 384-576** window, as the useful checkpoint region. The final checkpoint should probably not be presented as the best model. For the next run, I’d add early stopping on eval reward, inspect why completions become very short/empty, and check why `pg_clipfrac` never moves.