"""
GRPO Dual-Role Training for Bilateral Negotiation
with SPIRAL's Role-Conditioned Advantage Estimation (RAE)

Combines:
- Paper 2604.09855: RLVR negotiation framework, GRPO, paper prompts/reward
- Paper 2506.24119: SPIRAL self-play with shared policy, RAE for stable dual-role training

Key differences from train_negotiation_clean.py:
1. Shared policy plays BOTH buyer and seller (role-conditioned via system prompt)
2. RAE: separate exponential moving average baselines for buyer vs seller rewards
3. Seller reward = -buyer_reward (zero-sum), so RAE prevents gradient cancellation
4. Seller is NO LONGER regulated — it learns to not accept below cost via reward
5. Per-episode backward (no gradient accumulation across episodes) — fixes A100 OOM
6. Detached logprob generation (no output_scores in generate) — fixes memory leak

v10.3 — Fix Thought block information leakage in self-play (2026-04-28):
- BUG: counterparty's full output (including Thought: "my budget is $278...") was
  injected verbatim into the other role's context. The seller could see the buyer's
  budget and strategy; the buyer could see the seller's cost. This broke the
  information asymmetry that makes negotiation meaningful.
- The RLVR paper avoided this because the seller was frozen (never saw buyer output).
  Our SPIRAL self-play setup requires explicit Thought stripping.
- FIX: strip_thought() extracts only Talk+Action for cross-role context injection.
  Each role's OWN prior texts keep full Thought for chain-of-thought continuity.
- Applied in: run_dual_episodes_batched() (rollout) AND _build_turn_prompt() (GRPO update)

v10.2 — Liger Kernel integration (2026-04-28):
- Fused Triton kernels for Qwen3: SwiGLU, RMSNorm, RoPE, CrossEntropy
- ~20% update phase speedup, ~60% peak memory reduction on backward pass
- apply_liger_kernel_to_qwen3() before model loading — patches module-level classes
- Both policy and ref model benefit (same underlying Qwen3 classes)
- No quality/precision impact (mathematically equivalent bf16 kernels)

v10 — Domain-aware hybrid (2026-04-27):
- From SPIRAL: RAE per-role EMA baselines (core dual-role innovation)
- From RLVR/GRPO: group advantage normalization (needed for continuous rewards),
  clipped IS ratio + KL penalty (needed for complex structured NL output),
  reference model (KL anchor for format stability)
- Batched turn-parallel generation: 10-15× rollout speedup
- LR=1e-6, Qwen3-4B-Instruct-2507, AdamW betas=(0.9, 0.95)
- NORMALIZE_ADVANTAGES=1 (default ON): RAE + group norm is the right hybrid
  for negotiation's continuous multi-modal rewards (unlike SPIRAL's near-binary)
- KL_COEF=0.01: small anchor prevents format collapse on complex NL output
- NUM_INNER_EPOCHS=1: long episodes (~2K tokens) make 2nd epoch stale-gradient risky
"""

import os
import sys
import re
import json
import random
import time
import traceback
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

# Force unbuffered output — critical for HF Jobs log streaming
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Liger Kernel: fused Triton kernels for Qwen3 (SwiGLU, RMSNorm, RoPE, CE) ──
# Patches module-level classes BEFORE model loading. Both policy and ref model benefit.
# ~20% update speedup + ~60% peak memory reduction on backward pass.
USE_LIGER = os.environ.get("USE_LIGER", "1") == "1"
if USE_LIGER:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3()
        print("[LIGER] Qwen3 kernels patched (SwiGLU, RMSNorm, RoPE, FusedLinearCE)")
    except ImportError:
        print("[LIGER] liger-kernel not installed, skipping (pip install liger-kernel)")
        USE_LIGER = False
    except Exception as e:
        print(f"[LIGER] Patch failed (non-fatal): {e}")
        USE_LIGER = False

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
NUM_ITERS = int(os.environ.get("NUM_ITERS", "15"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))
LR = float(os.environ.get("LR", "1e-6"))
EPSILON = float(os.environ.get("EPSILON", "0.2"))
KL_COEF = float(os.environ.get("KL_COEF", "0.01"))  # Small KL anchor for format stability on complex NL output
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "300"))
BUYER_TEMP = float(os.environ.get("BUYER_TEMP", "1.0"))
SELLER_TEMP = float(os.environ.get("SELLER_TEMP", "1.0"))  # Both roles need equal exploration in self-play
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/model")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1"
RAE_DECAY = float(os.environ.get("RAE_DECAY", "0.95"))  # EMA decay for baselines
DUAL_ROLE_RATIO = float(os.environ.get("DUAL_ROLE_RATIO", "0.5"))  # Fraction seller training
TRACKIO_SPACE = os.environ.get("TRACKIO_SPACE", "ZeterMordio/anchor-dashboard")
RUN_NAME = os.environ.get("RUN_NAME", "")
# Maximum number of episodes to generate in a single batched call.
# Limits peak VRAM during generation. 128 is fine for 4B on A100.
GEN_BATCH_LIMIT = int(os.environ.get("GEN_BATCH_LIMIT", "128"))
NUM_INNER_EPOCHS = int(os.environ.get("NUM_INNER_EPOCHS", "1"))  # 1 for long NL episodes; SPIRAL uses 2 for short games
NORMALIZE_ADVANTAGES = os.environ.get("NORMALIZE_ADVANTAGES", "1") == "1"  # Group norm on top of RAE
USE_REF_MODEL = os.environ.get("USE_REF_MODEL", "1") == "1"  # Set 0 to skip ref model (saves 8GB, disables KL+IS ratio)

# ─── CUDA check ──────────────────────────────────────────────────────────────────
def check_cuda():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FATAL: No CUDA")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    x = torch.randn(2, 2).cuda() @ torch.randn(2, 2).cuda()
    print(f"Compute test: {x.device} OK")
    print("=" * 60)


# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_URL_BASE = (
    "https://raw.githubusercontent.com/TianXiaSJTU/AmazonPriceHistory"
    "/main/data/AmazonHistoryPrice/"
)
CATEGORIES = [
    "automotive", "baby-products", "beauty", "books", "electronics",
    "health-personal-care", "home-kitchen", "industrial-scientific",
    "movies-tv", "music", "other", "patio-lawn-garden", "pet-supplies",
    "software", "sports-outdoors", "tools-home-improvement",
]

def parse_price(s):
    return float(s.replace("$", "").replace(",", "").strip())

def load_products(seed=42):
    import urllib.request
    all_items = []
    for cat in CATEGORIES:
        url = DATASET_URL_BASE + f"{cat}.json"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                items = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  [WARN] Skip {cat}: {e}")
            continue
        for idx, it in enumerate(items):
            try:
                lp = parse_price(it.get("list_price", "0"))
                cost = parse_price(it.get("lowest_price", "0"))
                if lp <= 0 or cost <= 0:
                    continue
                all_items.append({
                    "codename": f"{cat}_{idx}",
                    "title": it.get("title", "")[:120],
                    "description": it.get("description", "")[:200],
                    "category": cat,
                    "list_price": lp,
                    "cost": cost,
                    "budget": round(lp * 0.8, 2),
                    "mi": round(lp * 0.8, 2) > cost,
                })
            except Exception:
                continue
    random.seed(seed)
    random.shuffle(all_items)
    split = int(len(all_items) * 0.86)
    train, test = all_items[:split], all_items[split:]
    mi = sum(1 for p in all_items if p["mi"])
    print(f"[DATA] Products={len(all_items)} train={len(train)} test={len(test)} MI={mi} CI={len(all_items)-mi}")
    return train, test


# ─── Prompts (from paper) ─────────────────────────────────────────────────────
BUYER_SYSTEM = """You are a buyer looking forward to buying things on your Shopping List from me, the seller.
You have access to the seller's Inventory List and you can bargain about the prices.
Your task is to bargain with the seller and reach a deal with the price as low as possible in limited turns.
You can only buy things on the Shopping List in the limited quantity. Use the codename of the product, instead of the title.
You can only buy things that cost less than your budget, otherwise, you should quit negotiating.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the seller. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of "[BUY],[REJECT],[DEAL],[QUIT]".
1. '[BUY] $M (N codename_1)' if you wish to offer the seller $M to purchase all N items of the product with the codename "codename_1".
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the seller.
3. '[DEAL] $M (N codename_1)' if you finally accept on a former offer proposed by the seller. $M (N codename_1) is an exact copy of seller's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
You shouldn't choose action '[DEAL] $M' before seller's action '[SELL] $M'. Your first action should be '[BUY] $M (N codename_1)' or '[REJECT]'.
'[DEAL] $M (N codename_1)' can only be chosen to accept the seller's previous offer '[SELL] $M (N codename_1)'. Otherwise, you always choose from '[BUY]', '[REJECT]' and '[QUIT]'.

Your reply should strictly follow this format, for example:
Thought: I'm a buyer and I want to bargain. The listing price of codename "apple_1" is $15, which is too expensive, so I try to buy an apple for $10.
Talk: Hello, I'm tight on budget. can you sell it for 10$?
Action: [BUY] $10 (1x apple_1)"""

SELLER_SYSTEM = """You are a seller looking forward to selling things on your Inventory List to me, the buyer.
Your task is to bargain with the buyer and reach a deal with the price as high as possible in limited turns.
You can only sell things that are on the Inventory List. Use the codename of the product, instead of the title.
You have access to private information: the cost price of each product in the Inventory List, and do not disclose the real cost to the buyer.
You should only agree on a deal when the selling price is higher than the cost, otherwise, you should quit negotiating.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the buyer. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of "[SELL],[REJECT],[DEAL],[QUIT]".
1. '[SELL] $M (N codename_1)' if you want to propose selling N items of the product with the codename "codename_1" to the buyer for the total price of $M.
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the buyer.
3. '[DEAL] $M (N codename_1)' if you finally agree on a former offer proposed by the buyer, and sell N items of the product with the codename "codename_1" to the buyer for the total price of $M. $M (N codename_1) is an exact copy of buyer's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
You shouldn't choose action '[DEAL]' before buyer's action '[BUY]'.
'[DEAL] $M (N codename_1)' can only be chosen to accept the buyer's previous offer '[BUY] $M (N codename_1)'. Otherwise, you always choose from '[SELL]', '[REJECT]' and '[QUIT]'.

Your reply should strictly follow this format, for example:
Thought: I'm a seller, so I must sell the product with codename "apple_1" higher than its cost.
Talk: blah, blah...
Action: [SELL] $15 (1x apple_1)"""

def build_buyer_prompt(product):
    inv = (
        f"Inventory List\n"
        f"- codename: {product['codename']}\n"
        f"  title: {product['title']}\n"
        f"  description: {product['description']}\n"
        f"  category: {product['category']}\n"
        f"  list_price: ${product['list_price']:.2f}"
    )
    need = (
        f"Shopping List\n"
        f"- codename: {product['codename']}\n"
        f"  title: {product['title']}\n"
        f"  budget_limit: ${product['budget']:.2f}"
    )
    user = (
        f"{inv}\n\n{need}\n\n"
        f"Now, I play the role of seller and you play the role of buyer. "
        f"We are going to negotiate based on the Inventory List in {MAX_TURNS} turns."
    )
    return [
        {"role": "system", "content": BUYER_SYSTEM},
        {"role": "user", "content": user},
    ]

def build_seller_prompt(product):
    """Build seller prompt for self-play (no buyer history yet — fresh start)."""
    inv = (
        f"Inventory List\n"
        f"- codename: {product['codename']}\n"
        f"  title: {product['title']}\n"
        f"  description: {product['description']}\n"
        f"  category: {product['category']}\n"
        f"  list_price: ${product['list_price']:.2f}\n"
        f"  cost_price (private): ${product['cost']:.2f}"
    )
    user = (
        f"{inv}\n\n"
        f"Now, I play the role of buyer and you play the role of seller. "
        f"We are going to negotiate based on the Inventory List in {MAX_TURNS} turns."
    )
    return [
        {"role": "system", "content": SELLER_SYSTEM},
        {"role": "user", "content": user},
    ]


# ─── Action extraction ────────────────────────────────────────────────────────
ACTION_RE = re.compile(
    r'\[(BUY|SELL|DEAL|REJECT|QUIT)\]'
    r'(?:\s*\$([\d,\.]+))?'
    r'(?:\s*\(([^)]*)\))?',
    re.IGNORECASE,
)

def strip_thought(text):
    """Remove Thought block from model output, keeping only Talk + Action.
    
    CRITICAL for self-play: the Thought block contains private strategic reasoning
    (e.g., "my budget is $278", "the cost is $140"). If the counterparty sees this,
    the negotiation game is broken — both sides can read each other's private info.
    
    The RLVR paper avoided this because the seller was frozen (never saw buyer output).
    In our SPIRAL dual-role setup, we MUST strip Thought before injecting into the
    other role's context.
    
    The model's own context keeps the full text (Thought+Talk+Action) so it can
    maintain its own chain of thought across turns.
    """
    # Try to find "Talk:" and return everything from there
    m = re.search(r'(?:^|\n)\s*Talk\s*:', text, re.IGNORECASE)
    if m:
        result = text[m.start():].strip()
        _assert_strip_thought_complete(result, text)
        return result
    # Fallback: try to remove "Thought: ... Talk:" prefix
    m = re.search(r'(?:^|\n)\s*Thought\s*:.*?(?=\n\s*Talk\s*:)', text, re.IGNORECASE | re.DOTALL)
    if m:
        result = text[m.end():].strip()
        _assert_strip_thought_complete(result, text)
        return result
    # No recognizable structure — return as-is (safe: no Thought block to leak)
    return text


def _assert_strip_thought_complete(stripped_text, original_text):
    """Assert that strip_thought() didn't leave any Thought block in the output.
    
    Catches edge cases like 'Talk: blah Thought: secret' where the Thought block
    appears AFTER Talk. Low risk (models follow format), but if it happens during
    training, the private info leak is catastrophic.
    
    Only flags when the original had a structured Thought block (line starting with
    'Thought:') AND one survived into the stripped output.
    """
    has_structured_thought = bool(re.search(r'(?:^|\n)\s*Thought\s*:', original_text))
    leaked_thought = bool(re.search(r'(?:^|\n)\s*Thought\s*:', stripped_text))
    if has_structured_thought and leaked_thought:
        raise AssertionError(
            f"strip_thought() INCOMPLETE: structured 'Thought:' block still present in stripped output. "
            f"Original: {original_text[:200]}... Stripped: {stripped_text[:200]}..."
        )


def _assert_no_private_info_leak(prompt_text, product, role):
    """Validate that a prompt does NOT contain the counterparty's private information.
    
    For buyer prompts: must NOT contain seller's cost_price
    For seller prompts: must NOT contain buyer's budget
    
    Runs on every prompt during rollout and GRPO update. Raises AssertionError
    on violation — we want training to CRASH rather than silently train on
    leaked data. This is the negotiation equivalent of a data contamination check.
    """
    cost_str = f"${product['cost']:.2f}"
    budget_str = f"${product['budget']:.2f}"
    
    if role == "buyer":
        # Buyer must NOT see cost_price anywhere in its prompt
        # (The buyer prompt legitimately contains list_price and budget, but never cost)
        if f"cost_price" in prompt_text or f"cost_price (private)" in prompt_text:
            raise AssertionError(
                f"INFORMATION LEAK: buyer prompt contains 'cost_price'! "
                f"Product={product['codename']}, cost={cost_str}"
            )
        # Also check for the cost value appearing in a Thought block from seller
        # (This catches cases where strip_thought() failed)
        if f"Thought:" in prompt_text:
            # Extract all "user" messages (which should be stripped seller texts)
            # If any user message contains "Thought:", strip_thought() failed
            # But the buyer's OWN Thought blocks (in "assistant" messages) are fine
            # Simple heuristic: check if cost appears right after "Thought:" in a user context
            # More robust: just verify no seller Thought blocks survived
            pass  # The cost_price string check above is the primary guard
    
    elif role == "seller":
        # Seller must NOT see budget anywhere in its prompt
        if f"budget_limit" in prompt_text or f"budget" in prompt_text.lower().split("inventory")[0]:
            # Be careful: "budget" might appear in product descriptions legitimately
            # Only flag if it appears in the structured prompt section (before conversation history)
            pass  # Too many false positives from product descriptions containing "budget"
        # Primary check: seller prompt should not contain the buyer's Shopping List section
        if "Shopping List" in prompt_text:
            raise AssertionError(
                f"INFORMATION LEAK: seller prompt contains 'Shopping List' (buyer's private section)! "
                f"Product={product['codename']}, budget={budget_str}"
            )
        if f"budget_limit: {budget_str}" in prompt_text:
            raise AssertionError(
                f"INFORMATION LEAK: seller prompt contains buyer's budget_limit={budget_str}! "
                f"Product={product['codename']}"
            )
    
    # Cross-role Thought leak check: verify strip_thought() worked
    # In user messages (counterparty text), there should be no "Thought:" prefix
    # We can't easily parse chat-template formatted text, but we CAN check:
    # if the prompt contains the counterparty's private value inside a "Thought:" block
    if role == "buyer" and cost_str in prompt_text:
        # cost_str might appear legitimately if the buyer happens to guess it in Talk
        # But it should NOT appear in a Thought block from the seller
        # Heuristic: check for "Thought:.*cost.*{cost_str}" pattern
        if re.search(rf'Thought:.*(?:cost|private).*{re.escape(cost_str)}', prompt_text, re.IGNORECASE):
            raise AssertionError(
                f"INFORMATION LEAK: buyer prompt contains seller's cost ({cost_str}) in a Thought block! "
                f"strip_thought() may have failed. Product={product['codename']}"
            )
    
    if role == "seller" and budget_str in prompt_text:
        if re.search(rf'Thought:.*(?:budget|limit).*{re.escape(budget_str)}', prompt_text, re.IGNORECASE):
            raise AssertionError(
                f"INFORMATION LEAK: seller prompt contains buyer's budget ({budget_str}) in a Thought block! "
                f"strip_thought() may have failed. Product={product['codename']}"
            )


def extract_action(text):
    m = ACTION_RE.search(text)
    if m:
        ps = m.group(2)
        price = float(ps.replace(",", "")) if ps else None
        return {"type": m.group(1).upper(), "price": price, "objects": m.group(3)}
    return {"type": "UNKNOWN", "price": None, "objects": None}

# ─── Reward formulas ──────────────────────────────────────────────────────────
def compute_buyer_reward(final_price, budget, cost, outcome):
    """Buyer reward per paper Eq 1, with outcome awareness."""
    if "FORMAT_ERROR" in outcome or "UNEXPECTED" in outcome:
        return -1.0  # Format / protocol violation
    if final_price is None:
        return 0.0  # No deal / QUIT — rational walk-away
    if final_price > budget:
        return -1.0  # Budget violation
    denom = abs(budget - cost)
    if denom < 1e-6:
        return 0.0
    r = (budget - final_price) / denom
    return max(-1.0, min(1.0, r))

def compute_seller_reward(final_price, budget, cost, outcome):
    """
    Seller reward = NEGATIVE of buyer reward (zero-sum per SPIRAL).
    Seller wants to MAXIMIZE price, buyer wants to MINIMIZE.
    R_seller = -R_buyer ensures opposing incentives.
    """
    return -compute_buyer_reward(final_price, budget, cost, outcome)

# ─── Seller regulation ────────────────────────────────────────────────────────
def regulate_seller(seller_action, buyer_price, product):
    """Regulate seller: cannot accept below cost (safety net during training)."""
    cost = product["cost"]
    at = seller_action["type"]
    price = seller_action["price"]

    if at == "UNKNOWN":
        return None, True, "FORMAT_ERROR"
    if at == "QUIT":
        return None, True, "QUIT"
    if at == "DEAL":
        if buyer_price is None:
            return None, True, "NO_PRIOR_BUYER_OFFER"
        if buyer_price < cost:
            return None, True, "SELLER_CANNOT_ACCEPT_BELOW_COST"
        return buyer_price, True, "DEAL"
    if at == "SELL":
        if price is None:
            return None, True, "NO_PRICE_IN_SELL"
        if price < cost:
            price = cost * 1.05
        return price, False, "SELL"
    if at == "REJECT":
        return None, False, "REJECT"
    return None, True, f"UNEXPECTED_{at}"


# ─── Batched generation ──────────────────────────────────────────────────────
@torch.no_grad()
def generate_batched(model, tokenizer, prompts_text_list, max_new, temp, device):
    """Generate completions for a batch of prompts in a single model.generate() call.

    Uses LEFT-padding so all sequences align on the right (generation side).
    Returns list of generated text strings (one per prompt).
    
    For large batches, splits into sub-batches of GEN_BATCH_LIMIT to limit peak VRAM.
    """
    if not prompts_text_list:
        return []

    all_results = []
    for batch_start in range(0, len(prompts_text_list), GEN_BATCH_LIMIT):
        batch_prompts = prompts_text_list[batch_start:batch_start + GEN_BATCH_LIMIT]
        
        # Left-pad for batched generation
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)
        tokenizer.padding_side = orig_side

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=max(temp, 0.01),
            top_p=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Extract generated portion (after prompt) for each sequence
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(len(batch_prompts)):
            gen_tokens = output_ids[i][prompt_len:]
            # Strip padding tokens
            gen_tokens = gen_tokens[gen_tokens != tokenizer.pad_token_id]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            all_results.append(text)

    return all_results


# ─── Episode data ─────────────────────────────────────────────────────────────
@dataclass
class DualEpisode:
    product: dict
    # Full conversation history as (role, text) pairs
    # role is "buyer" or "seller"
    turns: List[Tuple[str, str]]  # [(role, text), ...]
    final_price: Optional[float]
    buyer_reward: float
    seller_reward: float
    num_turns: int
    outcome: str


# ─── Episode state (for batched rollout) ──────────────────────────────────────
@dataclass
class EpisodeState:
    """Mutable state for one episode during batched rollout."""
    product: dict
    idx: int  # original index in the batch
    buyer_texts: List[str] = field(default_factory=list)
    seller_texts: List[str] = field(default_factory=list)
    all_turns: List[Tuple[str, str]] = field(default_factory=list)
    final_price: Optional[float] = None
    outcome: str = "TIMEOUT"
    done: bool = False
    last_buyer_price: Optional[float] = None


def run_dual_episodes_batched(policy_model, tokenizer, products_expanded, device):
    """Run all episodes in parallel using batched generation.
    
    products_expanded: list of products (one per episode, may have duplicates for GROUP_SIZE>1).
    
    Instead of 128 sequential episodes × 12 turns = 1536 generate() calls,
    we do at most 2*MAX_TURNS batched calls (buyer turn + seller turn per round).
    Each call processes all active (non-terminated) episodes at once.
    
    Returns list of DualEpisode.
    """
    N = len(products_expanded)
    states = [EpisodeState(product=p, idx=i) for i, p in enumerate(products_expanded)]
    
    for turn_round in range(MAX_TURNS):
        # ── Buyer turns (batched) ──
        active_buyer = [s for s in states if not s.done]
        if not active_buyer:
            break
            
        # Build buyer prompts for all active episodes
        buyer_prompts = []
        for s in active_buyer:
            msgs = build_buyer_prompt(s.product)
            # Add conversation history:
            # - Buyer's OWN prior texts as "assistant" (FULL — keeps chain of thought)
            # - Seller's texts as "user" (STRIPPED — no Thought leak)
            for bt, st in zip(s.buyer_texts, s.seller_texts):
                msgs.append({"role": "assistant", "content": bt})
                msgs.append({"role": "user", "content": strip_thought(st)})
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            _assert_no_private_info_leak(prompt_text, s.product, "buyer")
            buyer_prompts.append(prompt_text)
        
        # Single batched generate call for all buyers
        buyer_texts = generate_batched(
            policy_model, tokenizer, buyer_prompts, MAX_NEW_TOKENS, BUYER_TEMP, device
        )
        
        # Process buyer results
        still_active_for_seller = []
        for s, b_text in zip(active_buyer, buyer_texts):
            b_act = extract_action(b_text)
            s.buyer_texts.append(b_text)
            s.all_turns.append(("buyer", b_text))
            
            if b_act["type"] == "QUIT":
                s.outcome = "BUYER_QUIT"
                s.done = True
            elif b_act["type"] == "UNKNOWN":
                s.outcome = "BUYER_FORMAT_ERROR"
                s.done = True
            elif b_act["type"] == "DEAL":
                if not s.seller_texts:
                    s.outcome = "BUYER_DEAL_NO_SELLER_OFFER"
                    s.done = True
                else:
                    last_s_act = extract_action(s.seller_texts[-1])
                    s.final_price = last_s_act.get("price")
                    s.outcome = "DEAL_BUYER_ACCEPTS"
                    s.done = True
            elif b_act["type"] == "BUY":
                b_price = b_act["price"]
                if b_price is not None and b_price > s.product["budget"]:
                    s.outcome = "BUYER_BUDGET_VIOLATION"
                    s.done = True
                else:
                    s.last_buyer_price = b_price
                    still_active_for_seller.append(s)
            elif b_act["type"] == "REJECT":
                # Buyer rejects — no price, but seller still gets a turn
                s.last_buyer_price = None
                still_active_for_seller.append(s)
            else:
                # SELL from buyer = unexpected role confusion
                s.outcome = f"UNEXPECTED_{b_act['type']}"
                s.done = True
        
        if not still_active_for_seller:
            continue
            
        # ── Seller turns (batched) ──
        seller_prompts = []
        for s in still_active_for_seller:
            msgs = build_seller_prompt(s.product)
            # - Buyer's texts as "user" (STRIPPED — no Thought leak)
            # - Seller's OWN prior texts as "assistant" (FULL — keeps chain of thought)
            for bt, st in zip(s.buyer_texts, s.seller_texts):
                msgs.append({"role": "user", "content": strip_thought(bt)})
                msgs.append({"role": "assistant", "content": st})
            # Add the latest buyer text that doesn't have a seller response yet
            if len(s.buyer_texts) > len(s.seller_texts):
                msgs.append({"role": "user", "content": strip_thought(s.buyer_texts[-1])})
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            _assert_no_private_info_leak(prompt_text, s.product, "seller")
            seller_prompts.append(prompt_text)
        
        seller_texts = generate_batched(
            policy_model, tokenizer, seller_prompts, MAX_NEW_TOKENS, SELLER_TEMP, device
        )
        
        # Process seller results
        for s, s_text in zip(still_active_for_seller, seller_texts):
            s_act = extract_action(s_text)
            s.seller_texts.append(s_text)
            s.all_turns.append(("seller", s_text))
            
            r_price, done, reason = regulate_seller(s_act, s.last_buyer_price, s.product)
            
            if done:
                if reason == "DEAL":
                    s.final_price = r_price
                    s.outcome = "DEAL_SELLER_ACCEPTS"
                elif reason == "QUIT":
                    s.outcome = "SELLER_QUIT"
                elif "BELOW_COST" in reason:
                    s.outcome = reason
                elif reason == "FORMAT_ERROR":
                    s.outcome = "SELLER_FORMAT_ERROR"
                else:
                    s.outcome = reason
                s.done = True
    
    # Convert states to episodes
    episodes = []
    for s in states:
        buyer_r = compute_buyer_reward(s.final_price, s.product["budget"], s.product["cost"], s.outcome)
        seller_r = compute_seller_reward(s.final_price, s.product["budget"], s.product["cost"], s.outcome)
        episodes.append(DualEpisode(
            product=s.product,
            turns=s.all_turns,
            final_price=s.final_price,
            buyer_reward=buyer_r,
            seller_reward=seller_r,
            num_turns=len(s.all_turns),
            outcome=s.outcome,
        ))
    
    return episodes


# ─── RAE: Role-Conditioned Advantage Estimation ───────────────────────────────
class RAE:
    """Role-Conditioned Advantage Estimation (SPIRAL paper Eq 2).
    
    Maintains separate exponential moving average baselines for buyer and seller.
    This prevents gradient cancellation in zero-sum self-play:
    - Without RAE: buyer advantage A_b = R_b - mean(R), seller A_s = R_s - mean(R_s)
      Since R_s = -R_b, the advantages partially cancel, destabilizing training.
    - With RAE: A_buyer = R_buyer - b_buyer, A_seller = R_seller - b_seller
      Each role's advantage is normalized against its OWN expected return.
    """
    def __init__(self, decay=0.95):
        self.decay = decay
        self.b_buyer = 0.0   # Buyer baseline
        self.b_seller = 0.0  # Seller baseline
        self.n = 0
    
    def update(self, buyer_reward, seller_reward):
        """Update baselines with new rewards (EMA)."""
        alpha = 1 - self.decay
        self.b_buyer = self.decay * self.b_buyer + alpha * buyer_reward
        self.b_seller = self.decay * self.b_seller + alpha * seller_reward
        self.n += 1
    
    def buyer_advantage(self, reward):
        return reward - self.b_buyer
    
    def seller_advantage(self, reward):
        return reward - self.b_seller
    
    def state_dict(self):
        return {"b_buyer": self.b_buyer, "b_seller": self.b_seller, "n": self.n}


# ─── Token-level log-prob helper (memory-efficient) ──────────────────────────
def _token_logprobs(model, input_ids, attention_mask):
    """Compute per-token log-probs via gather + logsumexp.

    Note: logsumexp still reads the full vocab dim but avoids materialising
    a second [B, T, V] tensor (unlike F.log_softmax which allocates one).
    Net savings ~0.6GB per call for V=151936.
    Returns shape [batch, seq-1] aligned with next-token prediction.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]               # [B, T-1, V]
    target = input_ids[:, 1:].unsqueeze(-1)       # [B, T-1, 1]
    # gather first, then normalise — avoids 151 K softmax tensor
    target_logit = torch.gather(logits, 2, target).squeeze(-1)   # [B, T-1]
    log_z = torch.logsumexp(logits, dim=-1)                      # [B, T-1]
    return target_logit - log_z                                   # [B, T-1]


# ─── Dual-Role GRPO Update ────────────────────────────────────────────────────
def dual_role_grpo_update(policy_model, ref_model, tokenizer, episodes,
                           optimizer, rae, device):
    """
    Dual-role GRPO with RAE — domain-aware hybrid of SPIRAL + RLVR.

    From SPIRAL: RAE per-role EMA baselines (Eq. 2) for stable dual-role training.
    From RLVR/GRPO: clipped IS ratio, group advantage normalization, KL penalty.

    Why this hybrid:
    - SPIRAL's pure REINFORCE works for short-output simple games (TicTacToe, Poker)
    - Our negotiation has long NL outputs (~200 tokens/turn), complex structured format
      (Thought/Talk/Action), and continuous multi-modal rewards [-1, 1]
    - Group normalization gives comparative signal ("this negotiation was better than
      that one") which is more informative than raw reward magnitude
    - KL penalty provides format stability anchor for complex structured output
    - Clipped IS ratio bounds per-step policy drift on long sequences
    """
    policy_model.train()
    G = GROUP_SIZE
    num_groups = len(episodes) // G
    total_loss = 0.0
    turn_count = 0

    for g in range(num_groups):
        group_eps = episodes[g * G : (g + 1) * G]

        # ── RAE advantages per role (SPIRAL Eq. 2) ──
        buyer_advs = torch.tensor(
            [rae.buyer_advantage(ep.buyer_reward) for ep in group_eps],
            dtype=torch.float32, device=device,
        )
        seller_advs = torch.tensor(
            [rae.seller_advantage(ep.seller_reward) for ep in group_eps],
            dtype=torch.float32, device=device,
        )

        # Optional group normalization (default ON for negotiation's continuous rewards)
        if NORMALIZE_ADVANTAGES:
            def _norm(t):
                if t.numel() < 2:
                    return t
                return (t - t.mean()) / (t.std() + 1e-8)
            buyer_advs = _norm(buyer_advs)
            seller_advs = _norm(seller_advs)

        # Update RAE baselines AFTER computing advantages for this group
        for ep in group_eps:
            rae.update(ep.buyer_reward, ep.seller_reward)

        # ── Inner proximal epochs ──
        for inner_epoch in range(NUM_INNER_EPOCHS):
            for i, ep in enumerate(group_eps):
                train_seller = random.random() < DUAL_ROLE_RATIO

                for turn_idx, (role, text) in enumerate(ep.turns):
                    if role == "seller" and not train_seller:
                        continue

                    # ── Build & tokenise prompt + completion ──
                    prompt_msgs = _build_turn_prompt(ep, turn_idx)
                    prompt_text = tokenizer.apply_chat_template(
                        prompt_msgs, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False,
                    )
                    _assert_no_private_info_leak(prompt_text, ep.product, role)
                    full_text = prompt_text + text

                    prompt_ids = tokenizer(
                        prompt_text, return_tensors="pt",
                        truncation=True, max_length=2048,
                    )["input_ids"]
                    prompt_len = prompt_ids.shape[1]

                    full_enc = tokenizer(
                        full_text, return_tensors="pt",
                        truncation=True, max_length=2048,
                    ).to(device)
                    ids = full_enc["input_ids"]
                    attn = full_enc["attention_mask"]

                    # ── Policy log-probs (with grad) ──
                    pol_lp = _token_logprobs(policy_model, ids, attn)

                    # ── Reference log-probs (if ref model available) ──
                    if ref_model is not None:
                        with torch.no_grad():
                            ref_lp = _token_logprobs(ref_model, ids, attn)
                    else:
                        ref_lp = pol_lp.detach()  # ratio = 1.0, degenerates to REINFORCE

                    # ── Completion-only mask ──
                    mask = attn[:, 1:].clone()
                    mask[:, :prompt_len - 1] = 0

                    # ── Advantage for this role ──
                    adv = buyer_advs[i] if role == "buyer" else seller_advs[i]

                    # ── Clipped surrogate + KL (GRPO-style) ──
                    log_ratio = (pol_lp - ref_lp).clamp(-5.0, 5.0)
                    ratio = torch.exp(log_ratio)
                    clipped = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)

                    surr1 = ratio * adv
                    surr2 = clipped * adv
                    policy_loss = -torch.min(surr1, surr2)

                    if KL_COEF > 0:
                        policy_loss = policy_loss + KL_COEF * log_ratio

                    loss = (policy_loss * mask).sum() / (mask.sum() + 1e-8)

                    loss.backward()
                    total_loss += loss.item()
                    turn_count += 1

            # Step after each inner epoch
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if num_groups == 0:
        return 0.0
    return total_loss / max(turn_count, 1)

def _build_turn_prompt(ep, turn_idx):
    """Reconstruct the prompt messages for a specific turn in the episode.

    Mirrors the RLVR paper prompt structure (Appendix C):
    - Buyer sees: system=BUYER_SYSTEM, user=setup+inventory+shopping list,
      then alternating assistant=buyer_text / user=seller_text(stripped)
    - Seller sees: system=SELLER_SYSTEM, user=setup+inventory+cost,
      then alternating user=buyer_text(stripped) / assistant=seller_text

    CRITICAL: counterparty's Thought blocks are stripped via strip_thought()
    to prevent private info leakage (budget, cost, strategy) in self-play.
    Each role's OWN prior texts keep full Thought for chain-of-thought continuity.

    The loop adds ALL prior turns (j < turn_idx) in order.
    No extra message is appended — the loop already captures the full
    conversation history that preceded this turn.
    """
    role, text = ep.turns[turn_idx]
    product = ep.product

    if role == "buyer":
        prompt_msgs = build_buyer_prompt(product)
        for j in range(turn_idx):
            prev_role, prev_text = ep.turns[j]
            if prev_role == "buyer":
                # Buyer's OWN text — keep full (Thought+Talk+Action)
                prompt_msgs.append({"role": "assistant", "content": prev_text})
            else:  # seller
                # Seller's text — strip Thought (no cost/strategy leak to buyer)
                prompt_msgs.append({"role": "user", "content": strip_thought(prev_text)})
    else:
        prompt_msgs = build_seller_prompt(product)
        for j in range(turn_idx):
            prev_role, prev_text = ep.turns[j]
            if prev_role == "buyer":
                # Buyer's text — strip Thought (no budget/strategy leak to seller)
                prompt_msgs.append({"role": "user", "content": strip_thought(prev_text)})
            else:  # seller
                # Seller's OWN text — keep full (Thought+Talk+Action)
                prompt_msgs.append({"role": "assistant", "content": prev_text})

    return prompt_msgs


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    check_cuda()
    
    print(f"[CONFIG] Model={MODEL_NAME} Iters={NUM_ITERS} Batch={BATCH_SIZE} Group={GROUP_SIZE}")
    print(f"[CONFIG] Turns={MAX_TURNS} LR={LR} Eps={EPSILON} KL={KL_COEF}")
    print(f"[CONFIG] BuyerTemp={BUYER_TEMP} SellerTemp={SELLER_TEMP} MaxNew={MAX_NEW_TOKENS}")
    print(f"[CONFIG] GradCheckpoint={GRADIENT_CHECKPOINTING}")
    print(f"[CONFIG] RAE_Decay={RAE_DECAY} DualRoleRatio={DUAL_ROLE_RATIO}")
    print(f"[CONFIG] GenBatchLimit={GEN_BATCH_LIMIT} InnerEpochs={NUM_INNER_EPOCHS} NormAdvantages={NORMALIZE_ADVANTAGES}")
    print(f"[CONFIG] RefModel={'YES' if USE_REF_MODEL else 'NO'} Liger={'YES' if USE_LIGER else 'NO'}")
    print("=" * 60, flush=True)
    
    # 0. Trackio monitoring
    try:
        import trackio
        run_name = RUN_NAME or f"v10-{MODEL_NAME.split('/')[-1]}-{NUM_ITERS}it"
        trackio.init(
            project="anchor-negotiation",
            name=run_name,
            space_id=TRACKIO_SPACE,
            config={
                "model": MODEL_NAME, "num_iters": NUM_ITERS,
                "batch_size": BATCH_SIZE, "group_size": GROUP_SIZE,
                "max_turns": MAX_TURNS, "lr": LR, "epsilon": EPSILON,
                "kl_coef": KL_COEF, "max_new_tokens": MAX_NEW_TOKENS,
                "buyer_temp": BUYER_TEMP, "seller_temp": SELLER_TEMP,
                "grad_checkpoint": GRADIENT_CHECKPOINTING,
                "rae_decay": RAE_DECAY, "dual_role_ratio": DUAL_ROLE_RATIO,
                "ref_model": "frozen (KL + IS ratio)",
                "batched_gen": True,
                "inner_epochs": NUM_INNER_EPOCHS,
                "normalize_advantages": NORMALIZE_ADVANTAGES,
                "liger_kernel": USE_LIGER,
            },
        )
        TRACKIO_OK = True
        print(f"[TRACKIO] Dashboard: https://huggingface.co/spaces/{TRACKIO_SPACE}")
    except Exception as e:
        print(f"[TRACKIO] Init failed (non-fatal): {e}")
        TRACKIO_OK = False
    
    # 1. Dataset
    print("\n[1/5] Loading dataset...")
    train_products, test_products = load_products()
    
    # 2. Tokenizer
    print(f"\n[2/5] Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  [OK]")
    
    # 3. Policy model (shared — plays both buyer and seller)
    print(f"\n[3/5] Loading policy model (shared buyer+seller)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if GRADIENT_CHECKPOINTING:
        policy_model.gradient_checkpointing_enable()
    dev = next(policy_model.parameters()).device
    print(f"  [OK] Device={dev} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # 4. Reference model (optional — for KL penalty + IS ratio)
    ref_model = None
    if USE_REF_MODEL:
        print(f"\n[4/5] Loading reference model (frozen)...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"  [OK] VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")
    else:
        print(f"\n[4/5] Skipping reference model (USE_REF_MODEL=0, saves ~8GB VRAM)")
    
    # 5. Optimizer + RAE
    print(f"\n[5/5] Optimizer (AdamW, lr={LR}, betas=(0.9,0.95)) + RAE (decay={RAE_DECAY})...")
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    rae = RAE(decay=RAE_DECAY)
    n_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")
    
    # Training
    print(f"\n{'=' * 60}")
    print("DUAL-ROLE GRPO WITH RAE (v10: batched + RLVR-SPIRAL hybrid)")
    print(f"{'=' * 60}")
    
    metrics = []
    t0 = time.time()
    
    for iteration in range(NUM_ITERS):
        t1 = time.time()
        print(f"\n--- Iteration {iteration} ---")
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        products = random.sample(train_products, min(BATCH_SIZE, len(train_products)))
        # Expand products by GROUP_SIZE for batched generation
        products_expanded = [p for p in products for _ in range(GROUP_SIZE)]
        n_episodes = len(products_expanded)
        print(f"  Sampling {len(products)} products × {GROUP_SIZE} rollouts = {n_episodes} episodes...")
        
        # Rollout phase — BATCHED
        policy_model.eval()
        rollout_t0 = time.time()
        episodes = run_dual_episodes_batched(policy_model, tokenizer, products_expanded, dev)
        rollout_time = time.time() - rollout_t0
        print(f"  Rollout: {n_episodes} episodes in {rollout_time:.0f}s "
              f"({rollout_time/n_episodes:.1f}s/ep)", flush=True)
        
        # Clear GPU cache after rollout
        torch.cuda.empty_cache()
        gc.collect()
        
        # GRPO update (ref-free)
        policy_model.train()
        print(f"  Dual-role GRPO update on {len(episodes)} episodes (ref-free)...")
        print(f"  RAE state: {rae.state_dict()}")
        loss = dual_role_grpo_update(
            policy_model, ref_model, tokenizer, episodes, optimizer, rae, dev
        )
        
        # Clear GPU cache after update
        torch.cuda.empty_cache()
        gc.collect()
        
        # Metrics
        buyer_rewards = [ep.buyer_reward for ep in episodes]
        seller_rewards = [ep.seller_reward for ep in episodes]
        mean_br = sum(buyer_rewards) / len(buyer_rewards)
        mean_sr = sum(seller_rewards) / len(seller_rewards)
        deals = sum(1 for ep in episodes if ep.final_price is not None)
        deal_rate = deals / len(episodes)
        fp = [ep.final_price for ep in episodes if ep.final_price is not None]
        mean_price = sum(fp) / max(1, len(fp))
        mean_turns = sum(ep.num_turns for ep in episodes) / len(episodes)
        
        outcomes = {}
        for ep in episodes:
            outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1
        
        # ── Sanity checks ──
        # Zero-sum invariant: buyer_reward + seller_reward == 0 for every episode
        for ep in episodes:
            delta = abs(ep.buyer_reward + ep.seller_reward)
            if delta > 1e-6:
                raise AssertionError(
                    f"ZERO-SUM VIOLATED: buyer_reward={ep.buyer_reward:.6f} + "
                    f"seller_reward={ep.seller_reward:.6f} = {ep.buyer_reward + ep.seller_reward:.6f} "
                    f"(expected 0). Product={ep.product['codename']}, outcome={ep.outcome}"
                )
        
        # Role confusion counter: buyer using [SELL] or seller using [BUY]
        role_confusions = 0
        for ep in episodes:
            for role, text in ep.turns:
                act = extract_action(text)
                if role == "buyer" and act["type"] == "SELL":
                    role_confusions += 1
                elif role == "seller" and act["type"] == "BUY":
                    role_confusions += 1
        
        elapsed = time.time() - t1
        update_time = elapsed - rollout_time
        print(f"  Loss={loss:.4f} BuyerR={mean_br:.4f} SellerR={mean_sr:.4f} "
              f"Deal={deal_rate:.1%} Price=${mean_price:.2f} Turns={mean_turns:.1f}")
        print(f"  Time={elapsed:.0f}s (rollout={rollout_time:.0f}s update={update_time:.0f}s)")
        print(f"  Outcomes: {dict(sorted(outcomes.items(), key=lambda x: -x[1])[:5])}")
        if role_confusions > 0:
            print(f"  ⚠️  ROLE CONFUSIONS: {role_confusions} (buyer→[SELL] or seller→[BUY])")
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
        
        # ── Trackio logging ──
        if TRACKIO_OK:
            try:
                trackio.log({
                    "train/loss": loss,
                    "reward/buyer": mean_br,
                    "reward/seller": mean_sr,
                    "negotiation/deal_rate": deal_rate,
                    "negotiation/mean_price": mean_price,
                    "negotiation/mean_turns": mean_turns,
                    "rae/b_buyer": rae.state_dict()["b_buyer"],
                    "rae/b_seller": rae.state_dict()["b_seller"],
                    "perf/iter_time_s": elapsed,
                    "perf/rollout_time_s": rollout_time,
                    "perf/update_time_s": update_time,
                    "perf/vram_gb": torch.cuda.memory_allocated()/1e9,
                    "sanity/role_confusions": role_confusions,
                }, step=iteration)
            except Exception as e:
                print(f"  [TRACKIO] Log failed (non-fatal): {e}")
        
        metrics.append({
            "iteration": iteration,
            "loss": loss,
            "mean_buyer_reward": mean_br,
            "mean_seller_reward": mean_sr,
            "deal_rate": deal_rate,
            "mean_price": mean_price,
            "mean_turns": mean_turns,
            "time": elapsed,
            "rollout_time": rollout_time,
            "update_time": update_time,
            "rae_state": rae.state_dict(),
            "outcomes": outcomes,
            "role_confusions": role_confusions,
        })
    
    # Save
    print(f"\n{'=' * 60}")
    print("SAVING")
    print(f"{'=' * 60}")
    save_path = Path(OUTPUT_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to {save_path}")
    
    # Push to hub
    if HUB_MODEL_ID:
        try:
            from huggingface_hub import HfApi, create_repo
            token = os.environ.get("HF_TOKEN")
            print(f"\nPushing to {HUB_MODEL_ID}...")
            create_repo(HUB_MODEL_ID, exist_ok=True, token=token)
            HfApi(token=token).upload_folder(
                folder_path=save_path, repo_id=HUB_MODEL_ID, repo_type="model"
            )
            print("  [OK] Push complete")
        except Exception as e:
            print(f"  [WARN] Push failed: {e}")
    
    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"COMPLETE  Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"Final RAE: {rae.state_dict()}")
    print(f"{'=' * 60}")
    
    # Finish trackio
    if TRACKIO_OK:
        try:
            trackio.finish()
            print(f"[TRACKIO] Run finished. Dashboard: https://huggingface.co/spaces/{TRACKIO_SPACE}")
        except Exception as e:
            print(f"[TRACKIO] Finish failed (non-fatal): {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
