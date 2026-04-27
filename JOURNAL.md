# Anchor Negotiation — Engineering Journal

> Authored by: Anton Künzi
> Last updated: 2026-04-27 02:10 UTC
> Session with: ZeterMordio

This document is the single source of truth for all design decisions,
bug fixes, hyperparameters, memory budgets, and experimental results.
It replaces the old `engineering_notebook.md` which is archived for reference.

---

## Quick Reference

### Hyperparameters (canonical)

| Parameter | Toy Run 3 v7 | Real Run (planned) |
|-----------|-----------|-------------------|
| Model | Qwen3-4B-Instruct-2507 | Qwen3-8B |
| Iters | 15 | 40-60 |
| Batch (products) | 16 | 64 (paper) |
| Group (rollouts/product) | 8 | 8 |
| LR | 1e-6 | 1e-6 |
| Max turns | 6 | 6 |
| Max tokens/turn | 300 | 300 |
| Buyer temp | 1.0 | 1.0 |
| Seller temp | 1.0 | 1.0 (self-play) |
| KL penalty | 0.01 | 0.01 |
| RAE decay | 0.95 | 0.95 |
| Dual-role ratio | 0.5 | 0.5-1.0 |
| Clip epsilon | 0.2 | 0.2 |
| AdamW betas | (0.9, 0.95) | (0.9, 0.95) |
| Hardware | a100-large | a100-large |

### Verified Model IDs

| Model | Exists | Size | Notes |
|-------|--------|------|-------|
| `Qwen/Qwen3-4B` | ✅ | ~7.5GB bf16 | Instruct-merged. Toy Runs 1-3 v1-v6. |
| `Qwen/Qwen3-4B-Instruct-2507` | ✅ | ~7.5GB bf16 | **Toy Run 3 v7+.** Aug 2025, better IFEval. |
| `Qwen/Qwen3-8B` | ✅ | ~15GB bf16 | Real Run target. |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ✅ | ~56GB | Paper's exact model. |
| `Qwen/Qwen3-1.7B` | ✅ | ~3.4GB | Toy Run 1 successful. |

### Verified Dataset

- **Name:** AmazonHistoryPrice (NOT on HF Hub direct)
- **Source:** https://github.com/TianXiaSJTU/AmazonPriceHistory
- **Products:** 930 total
- **Split:** 802 train / 128 test  
- **MI / CI:** 886 MI / 44 CI
- **Per product:** title, description, list_price, cost, budget(=list×0.8)

### Memory Budget

Qwen3-4B dual-role on A100 (80GB):
- Policy model: ~8GB bf16 parameters
- Reference model: ~8GB (frozen, no grad)
- AdamW states: ~16GB (2× fp32 per bf16 param)
- Activations per turn: ~2GB (gradient checkpointing halves this)
- **Total operational: ~32GB** ✓ fits with headroom

Qwen3-8B would need ~64GB operational — fits on A100x1 but tight.
Real Run needs A100x4 (320GB) safe.

---

## Session 2026-04-26/27 — Toy Run 3 Dual-Role + RAE

### Context
Continuing from prior work: Toy Runs 1 & 2 (buyer-only, Qwen3-4B, pipeline validated
and convergence signal observed) on A100-large. Toy Run 3 was repeatedly failing
on A100 with silent crashes / no clear logs. User wants to combine RLVR paper
(2604.09855) negotiation framework with SPIRAL paper (2506.24119) self-play and
Role-Conditioned Advantage Estimation (RAE) to train a model that can negotiate
robustly REGARDLESS of who goes first, and eventually beat GPT-5.4 on a fraction
of the compute budget.

### Major Issues Found & Fixed

#### 1. Silent A100 Crash — GPU Memory Leak in `generate()`

- **Root cause:** `model.generate(output_scores=True, return_dict_in_generate=True)`
  creates a float32 tensor of shape [batch, gen_len, 151936] = ~55MB PER generation.
  In dual-role, each episode has ~6 turns × 2 roles. With full batch (16×8=128 episodes):
  that accumulates to GBs of score tensors per iteration, eventually OOMing on the
  80GB A100. The OOM happened silently (no Python traceback — just kernel kill).
- **Fix:** Removed `output_scores=True` entirely. `generate_turn()` now returns only
  the decoded text. Log-probs are recomputed later during GRPO backward pass by
  re-tokenising and doing a forward pass on the saved text.

#### 2. Loss Explosion (→10^17 in iteration 2)

- **Root cause 1:** Raw log-ratio `ratio = exp(tok_logprobs - ref_logprobs)` has
  no upper bound. After a GRPO step shifts weights, the policy can drift far from
  ref, making ratios -> infinity. The `torch.clamp(ratio, 0.8, 1.2)` only bounds
  `clipped` — `ratio` in `surr1` is still unbounded.
- **Fix 1:** Clamp the *log* ratio first: `(pol_lp - ref_lp).clamp(-5.0, 5.0)`.
  This caps ratio at ~148 and floor at ~0.007. Prevents NaN/inf completely.
- **Root cause 2:** Advantages were raw `reward - mean`, un-normalised. Big reward
  spread = large step sizes = destabilisation.
- **Fix 2:** Zero-mean, unit-std normalisation per role per group:
  `(t - mean) / (std + 1e-8)`. Standard GRPO trick, essential for stability.

#### 3. Seller Prompt Bug — Role Mapping Was Backwards

- **Root cause:** Original `build_seller_prompt(product, buyer_history_texts)` did
  `even=assistant, odd=user` on a flat list of all messages, but this is from the
  *buyer's* POV. From the seller's perspective, buyer messages should be "user"
  (the counterparty speaking TO the seller) and seller's own prior responses are
  "assistant".
- **Fix:** For self-play, restructured `run_dual_episode()` to build seller
  messages explicitly from `buyer_texts` (→ `"user"`) and `seller_texts` (→ `"assistant"`).

#### 4. `_build_turn_prompt` Double-Added Last Seller Message

- **Root cause:** The function had a loop adding all prior turns (j < turn_idx),
  then an *extra* if-block that re-added the last seller message for buyer turns.
  This caused the final seller message to appear TWICE in buyer prompts, confusing
  the model.
- **Fix:** Removed the redundant extra append. The loop handles all history.

#### 5. Per-Episode Optimizer Step Breaking GRPO

- **Root cause:** Stepping the optimizer after every single episode meant that
  within a GRPO group of 8 episodes, the first episode's update changed weights
  before processing episodes 2-8, invalidating their reference log-probs and group
  advantage computation.
- **Fix:** Moved `optimizer.step()` to after the FULL group of G episodes is
  processed. Gradient accumulation per-turn backward is valid; the *step* must be
  group-level.

#### 6. Full Vocab Softmax Memory Waste

- **Root cause:** `F.log_softmax(logits[:, :-1, :], dim=-1)` materialises a
  [batch, seq, 151936] tensor — ~1.1GB for 2048 tokens — EVERY forward pass
  (both policy and ref = 2.2GB per turn).
- **Fix:** `_token_logprobs()` gathers the target logits first, then computes
  log-prob via `logsumexp` on the fly. No full vocab tensor. Cuts ~1GB per call.

### Key Design Choices

- **Shared policy vs separate:** One model plays both buyer and seller. Role
  conditioning is purely via system prompt. This is SPIRAL's approach and is
  essential for self-play: as the buyer improves, the seller improves too,
  maintaining challenge level.
- **Zero-sum rewards:** `seller_reward = -buyer_reward`. This ensures symmetric
  competitive pressure. The paper's regulated seller (cannot accept below cost)
  is a leftover from their frozen-seller design; we keep it as a safety net but
  both sides get verifiable reward signals.
- **RAE (Role-Conditioned Advantage Estimation):** Per SPIRAL Eq 2, we maintain
  separate EMA baselines for buyer and seller. This prevents RAE from being
  biased by role asymmetry (e.g. first-mover advantage in negotiations).
- **Seller temperature = 1.0:** Originally matched paper's 0.7, but that was for
  a FROZEN seller (consistency). In self-play, BOTH roles need equal exploration
  (SPIRAL uses 1.0 for both). Reverted to 1.0.
- **DUAL_ROLE_RATIO = 0.5:** Fraction of seller turns that get trained on.
  Empirically: buyer always trains (it's the primary), seller trains 50% of
  the time. Prevents seller gradients from dominating early when buyer is more
  important for baseline metrics.

### Files

| File | Role |
|------|------|
| `train_negotiation_clean.py` | Original buyer-only script (Toy Runs 1&2) |
| `train_negotiation_dual_role.py` | Dual-role + RAE script (Toy Run 3+) |
| `JOURNAL.md` | This file — living log of decisions |

### Toy Run 3 Status — FAILED (Root Cause Diagnosed)

**Job IDs attempted:**
- `69eec939d70108f37ace0516` — STUCK at SCHEDULING 7+ hours, cancelled (Docker pytorch image)
- `69eeb7f6d70108f37ace04d5` — CANCELED (Docker pytorch image)
- `69ee4d86d70108f37ace02ba` — CANCELED (Docker pytorch image)
- `69ee4d61d2c8bd8662bd0109` — CANCELED (Docker pytorch image)
- `69ef3557d70108f37ace074c` — SCHEDULING forever (Docker pytorch image, REST API)
- `69ef5164d70108f37ace07e4` — SCHEDULING forever (Docker pytorch image, REST API)
- `69ef53d6d70108f37ace07f1` — SCHEDULING forever (Docker pytorch image, REST API)
- `69ef556bd70108f37ace07f5` — ✅ **RAN** 5.8 min, ERROR (`hf jobs uv run`, uv image)
- `69ef56f8d70108f37ace0803` — ⏳ SUBMITTED (`hf jobs uv run`, torch pinned to CUDA 12)

**Diagnosis (2026-04-27, updated ~13:00 UTC):**

#### ROOT CAUSE 1: The `pytorch/pytorch` Docker Image Causes Infinite SCHEDULING

**This was the primary culprit for the 7-hour stall and all "stuck in scheduling" jobs.**

All jobs using `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` as `dockerImage`
appeared stuck in SCHEDULING indefinitely. Analysis of 100 historical jobs shows:

- Jobs using the pytorch Docker image with large payloads (>5KB env) → stuck forever
- Jobs using lightweight images (uv, python:3.11, nvidia/cuda) → run promptly
- The one successful training job (`69eab541`, 1142s) used pytorch image but was
  submitted during a period of low demand

The `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` image is ~8GB. On HF Jobs infra,
pulling this image competes with GPU scheduling. When A100 demand is even moderate,
the combination of large image pull + GPU allocation creates a deadlock-like state
where the job shows "Job started" (image pull began) but never progresses to running.

**Evidence:** Switching to `ghcr.io/astral-sh/uv:python3.12-bookworm` (~200MB)
immediately resolved the scheduling hang. Job `69ef556b` went from SCHEDULING to
RUNNING within minutes and actually executed.

**Fix:** Use `hf jobs uv run` which uses the lightweight uv image. Dependencies
(torch, transformers, etc.) are installed via `uv` at runtime, which is fast (~60s)
and avoids the massive Docker image pull.

#### ROOT CAUSE 2: PyTorch CUDA Version Mismatch (uv auto-resolution)

When using `hf jobs uv run --with torch`, uv resolves the latest PyTorch (2.11.0+cu130,
CUDA 13.0). But HF Jobs A100 machines have NVIDIA driver 12090 (CUDA 12.x only).

Error: `CUDA initialization: The NVIDIA driver on your system is too old (found version 12090)`

**Fix:** Pin torch version: `--with 'torch>=2.6.0,<2.7'` to get a CUDA 12.4 build.

#### Additional Script Issues Fixed (from earlier analysis)

1. **`torch_dtype` deprecated** → changed to `dtype=torch.bfloat16`
2. **`torch.inference_mode()` risky with shared tensors** → changed to `torch.no_grad()`
3. **No output buffering control** → added `PYTHONUNBUFFERED=1` + `flush=True`
4. **No rollout progress** → added logging every 4 products
5. **No generation safety** → added `repetition_penalty=1.1`
6. **Inaccurate `_token_logprobs` docstring** → corrected

### Toy Run 3 — Correct Submission Method

The official way to submit HF Jobs is via the `hf` CLI (installed via `curl -LsSf https://hf.co/cli/install.sh | bash`):

```bash
hf jobs uv run \
    --flavor a100-large \
    --timeout 4h \
    --with 'torch>=2.6.0,<2.7' --with transformers --with accelerate --with trackio --with huggingface_hub \
    --secrets HF_TOKEN \
    --env BATCH_SIZE=16 \
    --env GROUP_SIZE=8 \
    --env MODEL_NAME=Qwen/Qwen3-4B \
    --env NUM_ITERS=15 \
    --env HUB_MODEL_ID=ZeterMordio/anchor-negotiation-dual-role \
    --env PYTHONUNBUFFERED=1 \
    --env RUN_NAME=toy3-qwen3-4b-15it \
    --env TRACKIO_SPACE=ZeterMordio/anchor-dashboard \
    --detach \
    train_negotiation_dual_role.py
```

Key points:
- Uses `uv` (not pip) for dependency resolution — much faster, lightweight image
- `ghcr.io/astral-sh/uv:python3.12-bookworm` image (~200MB) — **NOT** pytorch (~8GB)
- **Pin torch to <2.7** to avoid CUDA 13 builds (HF A100 driver is CUDA 12.x)
- `--secrets HF_TOKEN` properly injects the token from HF's secret store
- Script is uploaded as `LOCAL_FILES_ENCODED` env var (base64, handled by CLI)
- `--detach` returns immediately; logs via `hf jobs logs <ID> --follow`

### Real Run 4 Status

- **Status:** NOT YET LAUNCHED
- **Planned:** Qwen3-8B, 40–60 iters, a100x4
- **Depends on:** Toy Run 3 results
- **Current time estimate:** TBD

### Next Steps

1. Fix and re-launch Toy Run 3 with corrected job submission
2. Evaluate: does buyer performance degrade vs buyer-only?
3. If OK → proceed to dual-role Real Run (Qwen3-8B, 40 iters)
4. If role confusion persists: implement frozen-seller warmup
5. Add evaluation script: benchmark vs GPT-5.4, adversarial personas (RLVR §5)

---

## Archived: Prior Session Notes (from engineering_notebook.md)

### Toy Run 1
- **Status:** COMPLETED (17.6 min)
- **Result:** Pipeline works! Iter 0 had 37.5% deals, then format collapsed
  (100% BUYER_FORMAT_ERROR)
- **Analysis:** Format collapse expected for 1.7B — too small to maintain
  structured output under RL pressure

### Past Job IDs
- Toy Run 2: `69eabb91e8e12c6f0a675661` (buyer-only, Qwen3-4B, 15 iters)
- Toy Run 3: `69eeb7f6d70108f37ace04d5` (dual-role + RAE, Qwen3-4B, 15 iters)

### Known HF Jobs Quirks
- **Use `hf jobs uv run`, NOT the REST API** — handles auth, script upload, log streaming
- Pin `torch>=2.6.0,<2.7` to avoid CUDA 13 builds (HF A100 driver is 12090)
- `--secrets HF_TOKEN` injects from HF secret store (NOT `--env HF_TOKEN=xxx`)
- REST API `secrets` must NOT be passed as array — causes "expected record"
- `create_repo()` and `HfApi()` need explicit `token=` param

---

## Toy Run 3 — FINALLY RUNNING (2026-04-27 ~12:45 UTC)

**Job ID:** `69ef56f8d70108f37ace0803`
**Status:** ✅ RUNNING on A100-SXM4-80GB

### What finally made it work

| Change | Impact |
|--------|--------|
| `hf jobs uv run` instead of Docker REST API | **Unblocked scheduling** — uv image (~200MB) vs pytorch (~8GB) |
| `--with 'torch>=2.6.0,<2.7'` | Fixed CUDA mismatch (torch 2.11 needs CUDA 13, HF has 12.x) |
| `--secrets HF_TOKEN` | Proper auth for model download + trackio + hub push |
| `PYTHONUNBUFFERED=1` + `flush=True` | Live log streaming from Python (uv install phase still batched) |

### Observed Performance (Iteration 0)

| Metric | Value | Notes |
|--------|-------|-------|
| Rollout time (128 eps) | ~34 min | Sequential model.generate(), major bottleneck |
| Time per episode | ~16s | 12 generate calls × ~1.3s each |
| Time per generate call | ~1.3s | Qwen3-4B, 300 max_new_tokens, A100 |
| GRPO update time | TBD | Awaiting iter 0 completion |
| **Est. time per iteration** | **~40 min** | Rollout + update + cache clearing |
| **Est. total (15 iters)** | **~10 hours** | ⚠️ Exceeds 4h timeout! |
| **Est. total cost** | **~$25** | At $2.50/hr A100 |

### Hyperparameter & Cost Tracking

| Config | Batch | Group | Eps/Iter | Est. Iter Time | Est. Total (15it) | Est. Cost | Hardware |
|--------|-------|-------|----------|----------------|-------------------|-----------|----------|
| **Current (v6)** | 16 | 8 | 128 | ~40 min | ~10 hr | ~$25 | A100 $2.50/hr |
| Reduced batch | 8 | 8 | 64 | ~20 min | ~5 hr | ~$12.50 | A100 $2.50/hr |
| Reduced group | 16 | 4 | 64 | ~20 min | ~5 hr | ~$12.50 | A100 $2.50/hr |
| Reduced both | 8 | 4 | 32 | ~10 min | ~2.5 hr | ~$6.25 | A100 $2.50/hr |
| **Batched gen** (opt) | 16 | 8 | 128 | ~5-8 min | ~1.5-2 hr | ~$4-5 | A100 $2.50/hr |
| Batched + L40S | 16 | 8 | 128 | ~8-12 min | ~2-3 hr | ~$4-5 | L40S $1.80/hr |
| Batched + no ref | 16 | 8 | 128 | ~4-6 min | ~1-1.5 hr | ~$2.50-4 | A100 $2.50/hr |

### Key for Qwen3-8B Real Run (planned)

| Config | Batch | Group | Eps/Iter | Est. Iter Time | Est. Total (40it) | Est. Cost | Hardware |
|--------|-------|-------|----------|----------------|-------------------|-----------|----------|
| Sequential (current arch) | 64 | 8 | 512 | ~160 min | ~107 hr | ~$267 | A100 $2.50/hr |
| **Batched gen** | 64 | 8 | 512 | ~15-25 min | ~10-17 hr | ~$25-42 | A100 $2.50/hr |
| Batched + vLLM | 64 | 8 | 512 | ~8-12 min | ~5-8 hr | ~$12-20 | A100 $2.50/hr |

**Conclusion:** Batched generation is mandatory for the Real Run. Without it, Qwen3-8B at paper scale (batch=64, group=8) would cost ~$267 and take 4+ days.

---

## Optimization Analysis (2026-04-27)

### The Bottleneck

Rollout is **>85% of iteration time**. Current approach calls `model.generate()` 
sequentially 1,536 times per iteration (128 episodes × 12 turns). Each call is 
~1.3s but the A100 is severely underutilized — it's doing single-sequence decoding
at batch_size=1 while 80GB of VRAM sits mostly idle.

### Optimization 1: Batched Turn-Parallel Generation (HIGH IMPACT, MODERATE EFFORT)

**Core insight:** Episodes are independent of each other — only turns within an 
episode are sequential. We can batch all 128 buyer-turn-1 prompts into a single 
`model.generate()` call, then all 128 seller-turn-1, then buyer-turn-2, etc.

- **Before:** 1,536 sequential generate calls (~34 min)
- **After:** 12 batched calls (~2-4 min, depending on padding overhead)
- **Speedup:** ~10-15×
- **Effort:** Moderate refactor of `run_dual_episode()` → `run_dual_episodes_batched()`
- **Complications:** Variable-length responses need left-padding; some episodes 
  terminate early and need masking in subsequent turns.

### Optimization 2: Drop the Reference Model (HIGH IMPACT, TRIVIAL EFFORT)

SPIRAL (our RAE source paper) uses **no KL penalty** (`kl_loss_coefficient=0.0`,
`kl_penalty_coefficient=0.0`). They validated this works for self-play. We're
already at `KL_COEF=0.0`, so the ref model does nothing except consume:
- ~8 GB VRAM (half the model memory)
- ~50% of the GRPO update forward pass time

**Fix:** Remove ref_model entirely. Use `ratio = 1.0` (no importance sampling) 
or simply `policy_loss = -advantage * log_prob`. The clipping still prevents 
catastrophic updates.

**Risk:** Without KL anchor, policy might drift faster. But RAE baselines 
provide variance reduction, and clip_epsilon=0.2 bounds step size. SPIRAL 
shows this is fine for competitive self-play.

### Optimization 3: vLLM / Prefix Caching (HIGH IMPACT, HIGHER EFFORT)

Replace `model.generate()` with vLLM inference engine:
- **Prefix caching:** System prompt (~200 tokens) computed once, reused for all 
  128 episodes. Saves ~25% of prefill compute per generation.
- **PagedAttention:** Handles variable-length batches efficiently, no padding waste.
- **Continuous batching:** Better GPU utilization during decoding phase.
- **Colocate mode:** Runs in same process, shares GPU with training. Use 
  `vllm_enable_sleep_mode=True` to offload during GRPO update.

### Optimization 4: L40S Instead of A100 (COST SAVING)

If batched generation brings rollout under control, L40S (48GB, $1.80/hr) fits:
- 2× Qwen3-4B bf16 = 16 GB
- Optimizer states = 16 GB  
- Activations + cache = ~8 GB
- **Total: ~40 GB** fits in 48 GB

L40S has lower memory bandwidth (864 GB/s vs 2 TB/s) so generation is ~30-50% 
slower per token. But with batching, total wall time is dominated by compute 
not bandwidth. **Estimated 28% cost savings** with ~20% time increase.

### Optimization 5: Liger Kernel for GRPO Update (MODERATE IMPACT, TRIVIAL)

Fuses cross-entropy + logprob computations in the update phase. Reported 
20% throughput boost and 60% memory reduction on the forward/backward pass.
Just needs `pip install liger-kernel` and a one-line config change.

### Recommended Priority Order

1. **Batched generation** — 10-15× rollout speedup, mandatory for Real Run
2. **Drop ref model** — free 8GB VRAM + 50% update speedup, follows SPIRAL
3. **L40S hardware** — 28% cost savings with minimal time increase
4. **vLLM colocate** — additional 2-3× on top of batching
5. **Liger kernel** — easy 20% update phase speedup

---

## Toy Run 3 v6 Results & Collapse Analysis (2026-04-27 ~14:40 UTC)

**Job ID:** `69ef56f8d70108f37ace0803` — CANCELED after 4 iterations (collapse confirmed)

### Iteration-by-Iteration Results

| Iter | Deal Rate | BuyerR | SellerR | Loss | Turns | Time | Top Outcome |
|------|-----------|--------|---------|------|-------|------|-------------|
| 0 | **53.9%** | +0.018 | -0.018 | 0.167 | 4.1 | 2305s | 51 seller accepts, 37 budget violations |
| 1 | **0.0%** | -1.000 | +1.000 | 0.000 | 1.3 | 1352s | 92 BUYER_FORMAT_ERROR, 36 UNEXPECTED_BUY |
| 2 | 0.0% | -0.992 | +0.992 | 0.240 | 1.1 | 1392s | 116 FORMAT_ERROR |
| 3 | 0.0% | -1.000 | +1.000 | 0.000 | 1.0 | 1671s | 128 FORMAT_ERROR (100%) |

### Root Cause: LR=3e-5 Too Aggressive for Dense 4B

The RLVR paper used `lr=3e-5` with `Qwen3-30B-A3B-Instruct-2507` — a **MoE with only 3.3B active params**. 
In MoE, only 8/128 experts receive gradient per token, so most parameters get zero gradient. 
On our **dense** Qwen3-4B, ALL 4B params get gradient. So 3e-5 is effectively ~30× more destructive.

SPIRAL used `lr=1e-6` on the exact same dense Qwen3-4B family. That's 30× lower.

### v7 Fix: Three Changes

1. **Model → `Qwen3-4B-Instruct-2507`**: Aug 2025 post-training with massively improved IFEval.
   GPQA +20, AIME +28 pts vs original. Same size (7.5GB bf16), drop-in replacement.
2. **LR → `1e-6`**: SPIRAL's proven stable value for dense Qwen3-4B. 30× lower than v6.
3. **KL_COEF → `0.01`**: Small anchor to prevent format collapse. Both papers used KL=0,
   but they had either (a) a much bigger MoE model or (b) games with much simpler output format.
   Negotiation has complex structured output (Thought/Talk/Action) — a small KL helps.
4. **AdamW betas → (0.9, 0.95)**: Match SPIRAL's optimizer config exactly.

### Qwen3.6 Investigation — NOT SUITABLE

Both `Qwen3.6-27B` and `Qwen3.6-35B-A3B` are **vision-language** models (`qwen3_5` architecture),
NOT text-generation. They use hybrid DeltaNet + full attention layers with a vision encoder built in.
Architecture type is `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration`.

| Model | Architecture | Type | Params | Active | Notes |
|-------|-------------|------|--------|--------|-------|
| Qwen3.6-27B | qwen3_5 (DeltaNet hybrid) | Vision-Language | 27.8B | 27.8B dense | NOT suitable for text negotiation |
| Qwen3.6-35B-A3B | qwen3_5_moe (MoE + DeltaNet) | Vision-Language | 36.0B | ~3B active | NOT suitable — vision encoder + hybrid attention |

**Key insight:** "Qwen3.6" is a collection name, not a model series. The text-only models with improved
instruction following are the `-2507` suffix models: `Qwen3-4B-Instruct-2507` and `Qwen3-30B-A3B-Instruct-2507`.
No `Qwen3-8B-Instruct-2507` exists.

### GPU Sweep for Real Run (text-only models)

| Model | Params | bf16 | Policy+Ref VRAM | GPU | Cost/hr |
|-------|--------|------|-----------------|-----|---------|
| Qwen3-4B-Instruct-2507 | 4.0B | 7.5GB | ~16GB | a10g-large (24GB) | $2/hr |
| Qwen3-8B | 8.2B | 16.3GB | ~33GB | a100-large (80GB) | $4/hr |
| Qwen3-30B-A3B-Instruct-2507 | 30.5B | 56.9GB | ~115GB | a100x4 (320GB) | $16/hr |

### Real Run Recommendation

For the $37-70 budget:
- **Qwen3-8B** on **a100-large** ($4/hr) with batched generation
- 40-60 iterations at ~15 min/iter (batched) = 10-15 hours → $40-60
- LR=1e-6, same KL/RAE config as v7
---

## v8: Batched Generation + Reference-Free GRPO (2026-04-27)

### Changes

**Optimization 1: Batched turn-parallel generation**
- Replaced `run_dual_episode()` (sequential, 1 episode) with `run_dual_episodes_batched()` (all episodes in parallel)
- Each turn round: one batched `model.generate()` for all active buyers, then one for all active sellers
- Episodes that terminate early are masked out of subsequent turns
- Left-padding for variable-length prompts in batched calls
- Sub-batching via `GEN_BATCH_LIMIT=128` to cap peak VRAM
- **Expected speedup: 10-15× on rollout phase** (2-4 min vs 34 min)
- Bug fixed: buyer `[REJECT]` was falling through to `UNEXPECTED_REJECT` — now correctly routes to seller turn

**Optimization 2: Reference model removed**
- No separate frozen ref model → saves **8GB VRAM** (half of model memory)
- GRPO update does ONE forward pass per turn instead of TWO
- Within each group, `optimizer.step()` hasn't been called, so policy weights = rollout-time weights
- `ratio = π/π_old = 1.0` → clipped surrogate degenerates to weighted REINFORCE
- Loss = `-advantage * log_prob(completion)` — exactly what SPIRAL uses
- **Expected speedup: ~50% on GRPO update phase** (1 forward pass vs 2)
- KL_COEF kept as config but currently inert (KL between policy and itself = 0)

### Combined Expected Impact

| Phase | v6 (128 eps) | v8 (128 eps) | Speedup |
|-------|-------------|-------------|---------|
| Rollout | ~34 min | ~3-5 min | 7-10× |
| GRPO update | ~3 min (2 fwd/turn) | ~1.5 min (1 fwd/turn) | ~2× |
| **Total/iter** | **~38 min** | **~5-7 min** | **5-8×** |
| **15 iters** | **~9.5 hr** | **~1.2-1.7 hr** | - |
| **Est. cost** | ~$24 | **~$3-4** | 6-8× cheaper |

### VRAM Budget (v8, Qwen3-4B-Instruct-2507 on A100-80GB)

| Component | VRAM |
|-----------|------|
| Policy model (bf16) | ~8 GB |
| ~~Ref model (bf16)~~ | ~~8 GB~~ → **0 GB** |
| AdamW states (fp32) | ~16 GB |
| Batched generation KV cache | ~2-4 GB (128 seqs × ~2K tokens) |
| GRPO update activations | ~2 GB (grad checkpoint) |
| **Total** | **~28-30 GB** (was ~36 GB) |

**Headroom: ~50 GB free** — could increase GEN_BATCH_LIMIT or run larger models.
---

## v9: Paper-Faithful Implementation Audit (2026-04-27)

### Deep Paper Re-read: Three Discrepancies Found and Fixed

**Discrepancy 1: Double-normalization (FIXED)**
- SPIRAL Eq. 2: `A = R_p - b_EMA` — raw EMA-subtracted advantage, NO further normalization
- GRPO: `A = (R - mean) / std` — group-level normalization
- Our v8 code: `A = (R_p - b_EMA - mean) / std` — BOTH applied (hybrid from neither paper)
- **Fix:** Removed group-level normalization. Pure RAE advantages as in SPIRAL.

**Discrepancy 2: Single inner epoch (FIXED)**
- SPIRAL Table 6: `Inner proximal update epochs: 2`
- Our v8 code: 1 epoch per group
- **Fix:** Added `NUM_INNER_EPOCHS=2`. Each epoch: forward pass on same trajectories with current (updated) weights → backward → step. 2nd epoch uses shifted weights, so gradients differ.

**Discrepancy 3: KL=0.01 (FIXED)**
- RLVR paper Table 5: `KL penalty: 0` (explicit)
- SPIRAL Table 6: `KL loss coefficient: 0.0`, `KL penalty coefficient: 0.0` (explicit)
- Our v8 code: `KL_COEF=0.01` (was no-op anyway since ref-free)
- **Fix:** Set default to `0.0` to match both papers.

### What the Audit Confirmed as Correct

| Aspect | Paper Source | Our Implementation | Status |
|--------|-------------|-------------------|--------|
| Loss type | SPIRAL Eq. 3 | `-A * log_prob` (pure REINFORCE) | ✅ |
| EMA baselines | SPIRAL Eq. 2, α=0.95 | Per-role EMA, α=0.95 | ✅ |
| No IS ratio | SPIRAL (no π/π_old ratio) | No ratio | ✅ |
| No ref model | SPIRAL (KL=0, no ref) | Single model, no ref | ✅ |
| Optimizer | SPIRAL Table 6 | AdamW β=(0.9,0.95), wd=0.0 | ✅ |
| LR | SPIRAL 1e-6 for dense 4B | 1e-6 | ✅ |
| Grad clip | SPIRAL Table 6: 1.0 | 1.0 | ✅ |
---

## v10: Domain-Aware SPIRAL-RLVR Hybrid (2026-04-27)

### Why v9's Pure SPIRAL Settings Don't Transfer to Negotiation

SPIRAL trains on TicTacToe and Kuhn Poker — short outputs (~100 tokens), near-binary rewards (win/lose),
simple action format. Our negotiation has ~200 tokens/turn, continuous rewards [-1,1] with mass at
-1/0 boundaries, and complex Thought/Talk/Action structured output with price parsing.

| Setting | SPIRAL (simple games) | v9 (copied SPIRAL) | v10 (domain-aware) | Reasoning |
|---------|----------------------|--------------------|--------------------|-----------|
| **Advantage norm** | OFF (binary rewards) | OFF | **ON** | Continuous multi-modal rewards need comparative within-group signal |
| **KL penalty** | 0.0 (simple format) | 0.0 | **0.01** | Complex structured NL output needs format stability anchor |
| **Ref model** | None (KL=0) | None | **Frozen copy** | Needed for meaningful KL; we have VRAM headroom on A100 |
| **Inner epochs** | 2 (short episodes) | 2 | **1** | Long ~2K token episodes make 2nd epoch stale-gradient risky |
| **Loss** | REINFORCE | REINFORCE | **Clipped surrogate + KL** | Bounds per-step drift on long sequences |
| **RAE baselines** | ✅ | ✅ | ✅ | Core dual-role innovation, always keep |

### What We Keep From SPIRAL (the part that actually matters)
- RAE: per-role EMA baselines — this is THE contribution for dual-role training
- Self-play: shared policy plays both roles via system prompt conditioning
- Zero-sum rewards: R_seller = -R_buyer

### What We Keep From RLVR/GRPO (domain-appropriate for NL negotiation)
- Clipped IS ratio: bounds policy drift per step on long NL sequences
- Group advantage normalization: comparative signal for continuous reward distributions
- KL penalty (small): format stability for Thought/Talk/Action structure
- Frozen reference model: provides meaningful KL anchor

### All Configs Made Env-Var Overridable
- `NORMALIZE_ADVANTAGES=1` (ON by default, set 0 for pure SPIRAL)
- `KL_COEF=0.01` (set 0.0 for pure SPIRAL)
- `NUM_INNER_EPOCHS=1` (set 2 for SPIRAL's config)
- Everything else: same env vars as before





