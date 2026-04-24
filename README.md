# Anchor: Bilateral Negotiation via RLVR

Replication and extension of ["Measuring Bargaining Abilities of LLMs"](https://huggingface.co/papers/2604.09855) (paper 2604.09855).

**Goal:** Train a small open model (Qwen3-4B/8B) with GRPO to match or exceed GPT-5.4's negotiation performance on a tight compute budget (~$30–50).

## Architecture

- **Policy (buyer):** Qwen3-4B or Qwen3-8B, fully fine-tuned
- **Reference / Seller:** Same model, frozen weights
- **Training:** Custom multi-turn GRPO loop (not TRL — their GRPO doesn't support multi-turn environments)
- **Dataset:** AmazonHistoryPrice (930 products, bilateral price negotiation)
- **Reward:** Paper's formula — `R = (budget - P_final) / |budget - cost|`, clipped to [-1, 1]

## Files

| File | Purpose |
|------|---------|
| `train_negotiation_clean.py` | Canonical training script (self-contained, base64-safe for HF Jobs) |
| `engineering_notebook.md` | Living log of bugs, fixes, and design decisions |

## Usage

### Local Development

```bash
pip install transformers torch
python train_negotiation_clean.py
```

Environment variables control the run:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-4B` | Buyer/seller model |
| `NUM_ITERS` | `5` | Training iterations |
| `BATCH_SIZE` | `16` | Products sampled per iteration |
| `GROUP_SIZE` | `8` | Rollouts per product (GRPO group) |
| `MAX_TURNS` | `6` | Max negotiation turns |
| `LR` | `3e-5` | AdamW learning rate |
| `KL_COEF` | `0.0` | KL penalty (paper uses 0) |
| `GRADIENT_CHECKPOINTING` | `1` | Enable to reduce VRAM |
| `HUB_MODEL_ID` | — | HF Hub repo to push to |

### HF Jobs

Submit via the HF Jobs API. The script is designed to run in a Docker container with the `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` image.

## Key Design Decisions

1. **Custom GRPO loop** — TRL's `GRPOTrainer` lacks multi-turn environment support needed for buyer↔seller dialogue
2. **Full fine-tuning** — LoRA not used; 80GB A100 has sufficient VRAM for Qwen3-4B
3. **Per-episode backward** — Avoids OOM from accumulating gradients across all episodes
4. **Qwen3-4B → 8B scaling** — Toy runs on 1.7B validated the pipeline; real runs use 4B/8B

## Hardware Requirements

| Model | Min GPU | Recommended | Est. Runtime (40 iters) |
|-------|---------|-------------|------------------------|
| Qwen3-1.7B | a10g-large (24GB) | a10g-large | ~20 min |
| Qwen3-4B | l40sx1 (48GB) | a100-large (80GB) | ~2–4 hr |
| Qwen3-8B | a100-large (80GB) | a100x4 (320GB) | ~4–8 hr |

## Progress

- [x] Toy Run 1: Pipeline validation (Qwen3-1.7B, 5 iters)
- [ ] Toy Run 2: Convergence signal (Qwen3-4B, 15 iters) — in queue
- [ ] Toy Run 3: Dual-role training + RAE
- [ ] Real Run: 40–60 iters, Qwen3-8B

See `engineering_notebook.md` for detailed session logs.
