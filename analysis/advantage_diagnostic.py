"""Offline GRPO advantage-signal diagnostic (inference only; trains nothing).

WHAT IT MEASURES
GRPO turns rewards into advantages WITHIN each prompt's group of K completions:
    A_i = (r_i - mean(r)) / std(r).
If all K rewards are equal, std(r) = 0, every A_i = 0, and that group gives NO
gradient signal. We call such a group "degenerate". This script replays what a
GRPO rollout group looks like for a saved checkpoint -- sample K completions per
prompt at the training temperature, score them with the SAME reward functions
used in training -- and reports the fraction of degenerate groups and the spread
sigma_r, as a function of K.

WHY (exam I.3 diagnostic + I.4 Q1)
It is exactly the "fraction of degenerate groups in which all K rewards are equal"
/ "distribution of advantages" diagnostic the exam lists, and it explains the
baseline collapse mechanistically: as completions degenerate (e.g. empty/uniform),
sigma_r -> 0, the advantage signal vanishes, and training stalls.

REWARD-SET NOTE
We import REWARD_FNS from scripts/rewards.py, so the score is the exact
sum-of-rewards GRPO optimised. On this branch that is the baseline 4-term reward
(no length penalty); scoring every checkpoint with the SAME reward is a fair,
comparable measurement of "how much reward variation does this policy produce".
Response length is reported separately, so the length effect is still visible.

Example:
    python analysis/advantage_diagnostic.py \
        --checkpoint /home/basiakoch/content/ckpts/actor --step 500 \
        --run-name control_g8bs1 --num-prompts 64 --pool-k 8 --subsample-ks 2 4 8
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make the scripts/ modules (config, data, model, evaluate, rewards) importable
# from this file in analysis/.
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from config import (
    DATA_SOURCE, MAX_PROMPT_LENGTH, TEST_DATA_DIR,
    TEMPERATURE, TOP_K, TOP_P, TOTAL_GENERATION_STEPS,
)
from data import get_dataset
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer
from evaluate import restore_lora, generate
from rewards import REWARD_FNS, match_numbers
from tunix.generate import sampler as sampler_lib


def score_group(question, gold, completions):
    """Score K completions for one prompt with the training reward functions.

    Each REWARD_FN is called exactly as in training. Returns the per-completion
    total reward and a dict of per-component rewards. check_numbers prints a debug
    block per call, so we silence stdout around the calls to keep the run readable.
    """
    k = len(completions)
    prompts = [question] * k            # reward fns mostly ignore prompts
    answer = [gold] * k
    question_list = [question] * k      # check_numbers reads kwargs["question"]

    components = {}
    totals = [0.0] * k
    for fn in REWARD_FNS:
        with contextlib.redirect_stdout(io.StringIO()):
            scores = fn(prompts=prompts, completions=completions,
                        answer=answer, question=question_list)
        components[fn.__name__] = [float(s) for s in scores]
        totals = [t + float(s) for t, s in zip(totals, scores)]
    return totals, components


def group_stats(rewards):
    """Return (mean, std, degenerate, advantages) for one group of K rewards.

    std is the population std (0 exactly when all K rewards are equal). If the
    group is degenerate we return zero advantages and flag it, never dividing by 0.
    """
    arr = np.array(rewards, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())              # ddof=0; == 0 iff all rewards equal
    degenerate = std == 0.0
    if degenerate:
        advantages = [0.0] * len(rewards)
    else:
        advantages = [float((r - mean) / std) for r in rewards]
    return mean, std, degenerate, advantages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="actor checkpoint directory, e.g. /home/.../ckpts/actor")
    ap.add_argument("--step", type=int, default=0, help="checkpoint step; 0 = latest")
    ap.add_argument("--no-restore", action="store_true",
                    help="evaluate the base model (do not restore adapters)")
    ap.add_argument("--run-name", required=True, help="label used in output filenames")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--pool-k", type=int, default=8, help="completions sampled per prompt")
    ap.add_argument("--subsample-ks", type=int, nargs="+", default=[2, 4, 8],
                    help="group sizes to analyse (each must be <= --pool-k)")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--source", default=DATA_SOURCE, choices=["tfds", "kaggle"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", default="analysis/results")
    ap.add_argument("--figures-dir", default="analysis/figures")
    args = ap.parse_args()

    for k in args.subsample_ks:
        if k > args.pool_k:
            raise SystemExit(f"--subsample-ks contains {k} > --pool-k {args.pool_k}")
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # ---- load model + checkpoint (same sequence as evaluate.py main()) ----
    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)
    if args.no_restore:
        print("Evaluating base model (no adapter restored).")
    else:
        restore_lora(lora, args.checkpoint, None if args.step == 0 else args.step)
    sampler = sampler_lib.Sampler(
        transformer=lora, tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
        ),
    )

    # ---- fixed prompt set (data.py shuffles with seed 42, so this is reproducible) ----
    dataset = get_dataset(TEST_DATA_DIR, "test", args.source)
    prompts = []
    for ex in dataset:
        prompts.append((ex["question"], ex["answer"]))
        if len(prompts) >= args.num_prompts:
            break
    print(f"Loaded {len(prompts)} prompts; sampling pool-k={args.pool_k} completions each "
          f"at temperature={args.temperature}, seed base {args.seed}.")

    # ---- generate, score, record ----
    completions_path = os.path.join(args.results_dir, f"advantage_{args.run_name}_completions.jsonl")
    groups_path = os.path.join(args.results_dir, f"advantage_{args.run_name}_groups.csv")
    summary_path = os.path.join(args.results_dir, f"advantage_{args.run_name}_summary.json")

    comp_file = open(completions_path, "w")
    group_rows = []
    advantages_kmax = []                # for the advantage-distribution plot
    kmax = max(args.subsample_ks)

    for i, (question, gold) in enumerate(prompts):
        # K genuinely-different samples: one generate() per completion with a
        # distinct seed (we do not rely on the sampler's internal batch behaviour).
        pool = [generate(question, sampler, eos_tokens,
                         temperature=args.temperature, top_k=TOP_K, top_p=TOP_P,
                         seed=args.seed + i * args.pool_k + k)
                for k in range(args.pool_k)]

        totals, components = score_group(question, gold, pool)

        for k, text in enumerate(pool):
            pred = m.group(1) if (m := match_numbers.search(text)) is not None else None
            comp_file.write(json.dumps({
                "run_name": args.run_name,
                "checkpoint": args.checkpoint,
                "prompt_index": i,
                "question": question,
                "completion_index": k,
                "completion": text,
                "prediction": pred,
                "gold": gold,
                "reward_components": {name: vals[k] for name, vals in components.items()},
                "total_reward": totals[k],
                "response_length": len(text),
            }) + "\n")

        for K in args.subsample_ks:
            sub_rewards = totals[:K]            # first K of the pool = a size-K group
            sub_lengths = [len(t) for t in pool[:K]]
            mean, std, degenerate, advantages = group_stats(sub_rewards)
            group_rows.append({
                "run_name": args.run_name,
                "prompt_index": i,
                "K": K,
                "rewards": ";".join(f"{r:.3f}" for r in sub_rewards),
                "mean_reward": mean,
                "sigma_r": std,
                "degenerate": int(degenerate),
                "mean_length": float(np.mean(sub_lengths)),
                "max_length": int(max(sub_lengths)),
            })
            if K == kmax:
                advantages_kmax.extend(advantages)

        if (i + 1) % 10 == 0:
            print(f"  scored {i + 1}/{len(prompts)} prompts")

    comp_file.close()

    with open(groups_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    # ---- terminal summary + JSON ----
    summary = {"run_name": args.run_name, "checkpoint": args.checkpoint, "step": args.step,
               "num_prompts": len(prompts), "pool_k": args.pool_k,
               "temperature": args.temperature, "seed": args.seed, "per_k": {}}
    print("\n==== advantage-signal summary ====")
    print(f"run={args.run_name}  prompts={len(prompts)}  pool_k={args.pool_k}")
    for K in args.subsample_ks:
        rows = [r for r in group_rows if r["K"] == K]
        deg = float(np.mean([r["degenerate"] for r in rows]))
        sig = [r["sigma_r"] for r in rows]
        length = [r["mean_length"] for r in rows]
        summary["per_k"][str(K)] = {
            "degenerate_fraction": deg,
            "sigma_r_mean": float(np.mean(sig)),
            "sigma_r_median": float(np.median(sig)),
            "mean_response_length": float(np.mean(length)),
        }
        print(f"  K={K}: degenerate groups={deg*100:.1f}%  "
              f"sigma_r mean={np.mean(sig):.3f} median={np.median(sig):.3f}  "
              f"mean length={np.mean(length):.0f}")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plots ----
    base = f"advantage_{args.run_name}"

    def save(fig_name):
        plt.tight_layout()
        plt.savefig(os.path.join(args.figures_dir, f"{base}_{fig_name}.png"), dpi=200)
        plt.savefig(os.path.join(args.figures_dir, f"{base}_{fig_name}.pdf"))
        plt.close()

    fracs = [summary["per_k"][str(K)]["degenerate_fraction"] for K in args.subsample_ks]
    plt.figure(figsize=(6, 4)); plt.plot(args.subsample_ks, fracs, "o-")
    plt.xlabel("group size K"); plt.ylabel("fraction of degenerate groups")
    plt.title(f"Degenerate groups vs K ({args.run_name})"); plt.grid(alpha=0.3)
    save("degenerate_fraction_vs_k")

    sig_kmax = [r["sigma_r"] for r in group_rows if r["K"] == kmax]
    plt.figure(figsize=(6, 4)); plt.hist(sig_kmax, bins=20)
    plt.xlabel(f"within-group reward std sigma_r (K={kmax})"); plt.ylabel("number of prompts")
    plt.title(f"sigma_r distribution ({args.run_name})")
    save("sigma_r_hist")

    len_kmax = [r["mean_length"] for r in group_rows if r["K"] == kmax]
    plt.figure(figsize=(6, 4)); plt.hist(len_kmax, bins=20)
    plt.xlabel("mean response length per prompt (chars)"); plt.ylabel("number of prompts")
    plt.title(f"Response length ({args.run_name})")
    save("length_hist")

    plt.figure(figsize=(6, 4)); plt.hist(advantages_kmax, bins=30)
    plt.xlabel(f"normalized advantage A_i (K={kmax})"); plt.ylabel("count")
    plt.title(f"Advantage distribution ({args.run_name}); spike at 0 = degenerate groups")
    save("advantage_hist")

    print(f"\nwrote:\n  {completions_path}\n  {groups_path}\n  {summary_path}\n  figures in {args.figures_dir}/")


if __name__ == "__main__":
    main()
