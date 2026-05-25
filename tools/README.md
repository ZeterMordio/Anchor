# Tools

Small one-off utilities that support the project but are not training or evaluation entrypoints.

| File | Purpose |
|---|---|
| `backfill_pure_to_wandb.py` | Backfills completed pure-GRPO metrics from `ZeterMordio/anchor-negotiation-pure/metrics.json` into W&B. Keep as a utility because it is useful for historical dashboard repair, but it should not be confused with the active SDPO training scripts. |
| `launch_qwen35_fastpath_sdpo.py` | Cost-safe dense Qwen3.5 HF Jobs launcher. Requires a prebuilt fastpath image, keeps `a100-large`, `USE_LORA=0`, `CHECKPOINT_EVERY=10`, W&B per-iter logging, and a finite timeout. Dry-runs unless `--execute` is passed. |
| `qwen35_fastpath_canary.py` | GPU smoke for the Qwen3.5 fastpath image: checks `causal-conv1d`/`flash-linear-attention` imports, Qwen3.5 load, short generate, and tiny backward. |
