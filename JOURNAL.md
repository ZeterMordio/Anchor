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

| Parameter | Toy Run 3 | Real Run (planned) |
|-----------|-----------|-------------------|
| Model | Qwen3-4B | Qwen3-8B |
| Iters | 15 | 40-60 |
| Batch (products) | 16 | 64 (paper) |
| Group (rollouts/product) | 8 | 8 |
| LR | 3e-5 | 3e-5 |
| Max turns | 6 | 6 |
| Max tokens/turn | 300 | 300 |
| Buyer temp | 1.0 | 1.0 |
| Seller temp | 1.0 | 1.0 (self-play) |
| KL penalty | 0 | 0 |
| RAE decay | 0.95 | 0.95 |
| Dual-role ratio | 0.5 | 0.5-1.0 |
| Clip epsilon | 0.2 | 0.2 |
| Hardware | a100-large | a100x4 (planned) |

### Verified Model IDs

| Model | Exists | Size | Notes |
|-------|--------|------|-------|
| `Qwen/Qwen3-4B` | ✅ | ~7.5GB bf16 | Instruct-merged. Toy Run 3 current. |
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

### Toy Run 3 Status

- **Job ID:** `69eeb7f6d70108f37ace04d5`
- **Launched:** 2026-04-27 01:12 UTC
- **Hardware:** a100-large (80GB VRAM)
- **Status:** ⏳ **RUNNING** (started 02:04 UTC, ~52 min SCHEDULING)
- **Expected runtime:** ~2.5–4 hours for 15 iterations
- **Current time estimate:** Complete by ~04:30–06:00 UTC

### Real Run 4 Status

- **Status:** NOT YET LAUNCHED
- **Planned:** Qwen3-8B, 40–60 iters, a100x4
- **Depends on:** Toy Run 3 results
- **Current time estimate:** TBD

### Next Steps (after Toy Run 3 completes)

1. Evaluate: does buyer performance degrade vs buyer-only? If not → proceed
   to dual-role Real Run (Qwen3-8B, 40 iters).
2. If role confusion (UNEXPECTED_BUY >30% after iter 5) persists:
   implement frozen-seller warmup (buyer-only iters 0-5, then dual-role).
3. Add evaluation script: benchmark vs GPT-5.4, adversarial personas (RLVR §5).

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
- `secrets` must NOT be passed as array in REST API — causes "expected record"
- Correct: put `"HF_TOKEN": "auto"` in `environment` dict
- `create_repo()` and `HfApi()` need explicit `token=` param
- REST endpoint for job submission: `POST /api/jobs/{namespace}`
  **NOT** `/api/jobs` (404). Must include namespace in path.
- Field is `flavor` not `hardware`
- Field is `script` URL, not `spaceId` or `dockerImage` (unless Docker mode)

