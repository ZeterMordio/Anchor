# Tools

Small one-off utilities that support the project but are not training or evaluation entrypoints.

| File | Purpose |
|---|---|
| `backfill_pure_to_wandb.py` | Backfills completed pure-GRPO metrics from `ZeterMordio/anchor-negotiation-pure/metrics.json` into W&B. Keep as a utility because it is useful for historical dashboard repair, but it should not be confused with the active SDPO training scripts. |
