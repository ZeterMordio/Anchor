# Anchor Negotiation RLVR — Engineering Notebook

**Date:** 2025-04-23
**Project:** Replicate and improve upon paper 2604.09855 (RLVR for Bilateral Negotiation)
**Goal:** Train a small open model (Qwen3-4B/8B) with GRPO to beat GPT-5.4 in negotiation on a tight compute budget.

---

## Key Learnings from Research (Round 1)

### TRL GRPO Multi-Turn: `environment_factory`

TRL supports multi-turn agent training via `environment_factory`:
- Environment class with `reset()` method returning `None` or string
- Public methods (not `_`-prefixed) become tools callable by the model
- Each method needs Google-style docstring + typed args
- Tool loop runs up to `max_tool_calling_iterations` times
- `reward_funcs` receives `environments` (list of env instances), returns list of floats

Critical: For Qwen3, **must set** `chat_template_kwargs={"enable_thinking": False}` — all TRL examples with Qwen3 do this.

### Qwen3 Model IDs (Verified)

| Model | Status | Size | Notes |
|-------|--------|------|-------|
| `Qwen/Qwen3-4B` | ✅ Exists | 7.5GB | Instruct-merged (no separate -Instruct) |
| `Qwen/Qwen3-8B` | ✅ Exists | 15.3GB | Instruct-merged |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ✅ Exists | 56.9GB | Paper's exact model |
| `Qwen/Qwen3-0.6B` | ✅ Exists | ~1.2GB | Failed tool-use in previous run |
| `Qwen/Qwen3-1.7B` | ✅ Exists | ~3.4GB | Used in TRL Game 2048 example |

Previous run bug: `Qwen/Qwen3-4B-Instruct` is a 404 — the correct ID is `Qwen/Qwen3-4B`.

### GRPOConfig Parameters (TRL)

Key params for negotiation:
- `chat_template_kwargs={"enable_thinking": False}` — REQUIRED
- `scale_rewards="none"` — matches paper (raw clipping, no normalization)
- `loss_type="dr_grpo"` — eliminates length bias
- `max_tool_calling_iterations=6` — 6 turns max per paper
- `max_completion_length=512` — per-turn completion
- `num_generations=8` — group size G
- `use_vllm=True, vllm_mode="server"` — for larger models

### Paper's Negotiation Protocol

From logs and paper methodology:

**Structured output format:**
```
<REASONING> ... strategic thinking ... </REASONING>
<DIALOGUE> ... text to say to seller ... </DIALOGUE>
<ACTION> [BUY] $X | [SELL] $X | [DEAL] $X | [REJECT] | [QUIT] </ACTION>
```

**Reward function (buyer):**
```
R = (budget - P_final) / |budget - cost|    clipped to [-1, 1]
Successful deal (MI, budget > cost): R ∈ [0, 1]
Deal in CI (budget < cost):          R ∈ [-1, 0]
Deadlock / QUIT:                     R = 0.0
Budget violation:                    R = -1.0
Format violation:                    R = -1.0
```

**Seller regulation:**
- Seller cannot accept below its cost
- Seller is "regulated" — constrained to not violate its floor

### Dataset: AmazonHistoryPrice

- Source: GitHub `TianXiaSJTU/AmazonPriceHistory`
- NOT on HF Hub
- 930 products, 18 categories
- 802 train / 128 test split
- 886 MI / 44 CI scenarios
- Per product: title, description, list_price, seller_cost (= lowest historical), buyer_budget (= 80% of list)
- Price range: $5 – $4,300

### Memory Budgeting (Lesson from Previous Run)

For GRPO with Qwen3-4B (bf16):
- Policy model: ~7.5GB
- Reference model: ~7.5GB
- vLLM (if used): ~7.5GB + kv cache (~2-4GB)
- Activations: ~2-4GB (depends on batch)
- Optimizer (LoRA): ~0.5-2GB
- **Total worst case: ~27GB**

l40sx1 (48GB) is comfortable for Qwen3-4B.
t4-small (16GB) is too tight for both policy + reference + activations.
For toy validation, we can use `Qwen/Qwen3-1.7B` (~3.4GB × 2 = ~7GB) on t4-small with gradient checkpointing.

### Toy Run Strategy

**Toy Run 1:** Pipeline validation with Qwen3-1.7B on t4-small
- 3-5 iterations, tiny batch
- Verify: dataset loads, env factory works, reward computes, GRPO updates
- Expected: no crashes, loss decreases (or at least doesn't crash)
- Cost: ~$2-4, 1-2 hours

**Toy Run 2:** Convergence signal with Qwen3-4B on l40sx1
- 15-20 iterations
- Look for: Phase 1 anchoring behavior, Phase 2 deadlock dip, Phase 3 recovery
- Cost: ~$5-8, 3-4 hours

**Toy Run 3 (optional):** Dual-role with Qwen3-4B on l40sx1
- 15-20 iterations, 50/50 buyer/seller
- Evaluate if buyer performance degrades
- Cost: ~$5-8, 3-4 hours

**Real Run:** Qwen3-8B on a100-large
- 40-60 iterations
- Cost: ~$25-40, 10-16 hours

---

## Bugs Encountered & Fixes (This Session)

### SSH Key Issue
- First key got lost when sandbox refreshed
- Had to regenerate and request deploy key again
- Fix: Configured `~/.ssh/config` to use specific IdentityFile
- Waiting for user to update deploy key on GitHub

### GitHub Repo Access
- `git@github.com: Permission denied (publickey)`
- Deploy key not yet updated on `ZeterMordio/Anchor`
- Working locally in `/app/anchor_negotiation/`, will push once access restored

---

## Bugs Encountered & Fixes (This Session — Toy Runs)

### Toy Run 1: Qwen3-1.7B on a10g-large (24GB)
- **Status:** COMPLETED (17.6 min)
- **Result:** Pipeline works! Iter 0 had 37.5% deals, then format collapsed (100% BUYER_FORMAT_ERROR)
- **Analysis:** Format collapse is expected for 1.7B — too small to maintain structured output under RL pressure
- **Action:** Reverted all format/KL fixes — not needed for larger models

### HF Jobs API: `secrets` field
- `secrets` must be an array of strings like `["HF_TOKEN"]` — not an object
- But passing it as `"secrets": ["HF_TOKEN"]` in the JSON payload caused `Invalid input: expected record, received array`
- **Fix:** Just put `"HF_TOKEN": "auto"` in `environment` dict — HF Jobs auto-injects it
- BUT: `create_repo()` and `HfApi()` in the script still got 401 because they don't read env var automatically
- **Fix:** Pass `token=os.environ.get("HF_TOKEN")` explicitly to both calls

### Toy Run 2: Qwen3-4B on a100-large (80GB)
- **Status:** SUBMITTED (Job ID: `69eabb91e8e12c6f0a675661`)
- **Config:** 15 iters, batch=16, group=8, LR=3e-5, KL=0.0
- **Expected:** ~2-4 hours, should show 4-phase convergence pattern

### H200 Issue (GitHub #4128)
- H200 nodes have NVIDIA Fabric Manager stuck at "In Progress" → CUDA Error 802
- **Confirmed:** Waiting 30-60s after `nvidia-smi` does NOT fix it
- **Workaround:** Use a10g-large or a100-large instead
- a10g-large (24GB, Ampere) works with `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

## Next Steps
1. Monitor Toy Run 2 (check logs in ~2 hours)
2. If successful: analyze metrics for 4-phase pattern
3. If successful: proceed to Toy Run 3 (dual-role) or Real Run (Qwen3-8B)
4. Push final trained model to HF Hub (token fix now in place)
