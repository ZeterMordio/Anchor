#!/usr/bin/env python3
"""Backfill completed pure negotiation GRPO metrics from HF Hub metrics.json to W&B."""

import json
import os
from collections import Counter
from pathlib import Path

import wandb
from huggingface_hub import hf_hub_download

REPO_ID = "ZeterMordio/anchor-negotiation-pure"
JOB_ID = "6a0538b5e48bea4538b9c3cf"
PROJECT = os.environ.get("WANDB_PROJECT", "anchor-negotiation-pure")
ENTITY = os.environ.get("WANDB_ENTITY", "chalk")
RUN_ID = os.environ.get("WANDB_RUN_ID", "pure-qwen3-4b-42it-backfill")
RUN_NAME = os.environ.get("RUN_NAME", "pure-qwen3-4b-42it-backfill")


def main():
    metrics_path = hf_hub_download(repo_id=REPO_ID, repo_type="model", filename="metrics.json")
    metrics = json.loads(Path(metrics_path).read_text())
    if not metrics:
        raise RuntimeError("No metrics found to backfill")

    config = {
        "method": "negotiation_pure_grpo_backfill",
        "source": "hf_hub_metrics_json",
        "source_repo": REPO_ID,
        "source_job_id": JOB_ID,
        "source_job_url": f"https://huggingface.co/jobs/ZeterMordio/{JOB_ID}",
        "source_model_url": f"https://huggingface.co/{REPO_ID}",
        "backfill": True,
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "seller_model": "Qwen/Qwen3-4B-Instruct-2507",
        "num_iters": 42,
        "batch_size": 16,
        "group_size": 8,
        "episodes_per_iter": 128,
        "max_turns": 6,
        "max_new_tokens": 300,
        "lr": 1e-6,
        "kl_coef": 0.01,
        "buyer_temp": 1.0,
        "seller_temp": 0.7,
        "checkpoint_every": 10,
        "gen_batch_limit": 128,
        "gradient_checkpointing": True,
        "hardware": "a100-large",
        "original_monitoring": "trackio",
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=RUN_ID,
        name=RUN_NAME,
        group="pure-qwen3-4b",
        job_type="backfill",
        tags=["pure", "grpo", "negotiation", "backfill", "qwen3-4b"],
        config=config,
        resume="allow",
    )

    all_outcomes = sorted({k for row in metrics for k in row.get("outcomes", {})})
    for row in metrics:
        step = int(row["iteration"])
        outcomes = row.get("outcomes", {}) or {}
        log = {
            "train/loss": row.get("loss"),
            "reward/buyer": row.get("mean_reward"),
            "negotiation/deal_rate": row.get("deal_rate"),
            "negotiation/mean_price": row.get("mean_price"),
            "negotiation/mean_turns": row.get("mean_turns"),
            "negotiation/first_offer_ratio": row.get("first_offer_ratio") or 0.0,
            "negotiation/price_overshoot_rate": row.get("price_overshoot_rate"),
            "perf/iter_time_s": row.get("time"),
            "perf/rollout_time_s": row.get("rollout_time"),
            "perf/update_time_s": row.get("update_time"),
            "perf/vram_gb": row.get("vram_current_gb"),
            "perf/vram_peak_gb": row.get("vram_peak_gb"),
            "sanity/role_confusions": row.get("role_confusions"),
        }
        for key in all_outcomes:
            log[f"outcomes/{key}"] = outcomes.get(key, 0)
        wandb.log(log, step=step)

    final = metrics[-1]
    best_reward = max(metrics, key=lambda r: r.get("mean_reward", float("-inf")))
    best_deal = max(metrics, key=lambda r: r.get("deal_rate", float("-inf")))
    total_outcomes = Counter()
    for row in metrics:
        total_outcomes.update(row.get("outcomes", {}) or {})

    summary = run.summary
    summary["backfill/source_repo"] = REPO_ID
    summary["backfill/source_job_id"] = JOB_ID
    summary["backfill/rows"] = len(metrics)
    summary["final/iteration"] = final.get("iteration")
    summary["final/reward"] = final.get("mean_reward")
    summary["final/deal_rate"] = final.get("deal_rate")
    summary["best_reward/value"] = best_reward.get("mean_reward")
    summary["best_reward/iteration"] = best_reward.get("iteration")
    summary["best_deal_rate/value"] = best_deal.get("deal_rate")
    summary["best_deal_rate/iteration"] = best_deal.get("iteration")
    summary["total/deals"] = total_outcomes.get("DEAL_BUYER_ACCEPTS", 0) + total_outcomes.get("DEAL_SELLER_ACCEPTS", 0)
    summary["total/buyer_format_errors"] = total_outcomes.get("BUYER_FORMAT_ERROR", 0)

    run.finish()
    print(f"Backfilled {len(metrics)} rows to {run.url}")


if __name__ == "__main__":
    main()
