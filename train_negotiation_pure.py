"""
Pure Negotiation-RLVR training for bilateral price negotiation.

This script implements the buyer-only setup from paper 2604.09855
("Instructing LLMs to Negotiate using Reinforcement Learning with
Verifiable Rewards") while preserving non-conflicting engineering fixes from
our SPIRAL/RLVR hybrid script.

Pure paper structure:
- Trainable buyer policy.
- Frozen seller / reference model as the environment counterparty.
- Buyer always starts; only buyer turns receive policy-gradient updates.
- Seller is regulated: it cannot accept/propose below private cost.
- Verifiable buyer reward: R = (budget - final_price) / |budget - cost|,
  clipped to [-1, 1]; no-deal/quit/timeout = 0; buyer format/budget errors = -1.

Kept from the newer SPIRAL script where non-conflicting:
- Batched turn-parallel generation for rollout speed.
- Thought stripping before cross-role context injection (paper §3.1 hidden scratchpad).
- Runtime private-information leak guards.
- Memory-efficient token log-prob computation (gather + logsumexp; no full softmax tensor).
- Clamped log-ratio before exp to prevent PPO/GRPO loss explosions.
- Group-level advantage normalization for continuous negotiation rewards.
- Small KL/reference anchor by default for dense 4B format stability (env-overridable).
- Liger kernel optional integration.
- Trackio metrics + alerts.
- Periodic HF Hub branch checkpoints.

Explicitly removed from SPIRAL:
- Shared policy self-play.
- Seller training.
- Seller reward = -buyer reward.
- Role-conditioned advantage estimation (RAE).
- Dual-role ratio / per-role baselines.

Default initial run is configured but NOT launched here:
- NUM_ITERS=42, BATCH_SIZE=16, GROUP_SIZE=8, a100-large recommended.
"""

import gc
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# HF Jobs log streaming: avoid buffered multi-minute stalls.
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Liger Kernel: optional fused Triton kernels for Qwen3 ────────────────────
USE_LIGER = os.environ.get("USE_LIGER", "1") == "1"
if USE_LIGER:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3()
        print("[LIGER] Qwen3 kernels patched (SwiGLU, RMSNorm, RoPE, FusedLinearCE)")
    except ImportError:
        print("[LIGER] liger-kernel not installed, skipping")
        USE_LIGER = False
    except Exception as e:
        print(f"[LIGER] Patch failed (non-fatal): {e}")
        USE_LIGER = False


# ─── Config ──────────────────────────────────────────────────────────────────
# Model choice: paper exact model is Qwen/Qwen3-30B-A3B-Instruct-2507. For the
# comparable small-scale run requested here we use the verified dense 4B model
# used in the SPIRAL test run; override MODEL_NAME for the paper MoE model.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
SELLER_MODEL_NAME = os.environ.get("SELLER_MODEL_NAME", MODEL_NAME)
NUM_ITERS = int(os.environ.get("NUM_ITERS", "42"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))
LR = float(os.environ.get("LR", "1e-6"))
EPSILON = float(os.environ.get("EPSILON", "0.2"))
KL_COEF = float(os.environ.get("KL_COEF", "0.01"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "300"))
BUYER_TEMP = float(os.environ.get("BUYER_TEMP", "1.0"))
SELLER_TEMP = float(os.environ.get("SELLER_TEMP", "0.7"))  # paper Table 5
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/model")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1"
GEN_BATCH_LIMIT = int(os.environ.get("GEN_BATCH_LIMIT", "128"))
NUM_INNER_EPOCHS = int(os.environ.get("NUM_INNER_EPOCHS", "1"))
NORMALIZE_ADVANTAGES = os.environ.get("NORMALIZE_ADVANTAGES", "1") == "1"
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "10"))
SEED = int(os.environ.get("SEED", "42"))
TRAIN_SPLIT_SIZE = int(os.environ.get("TRAIN_SPLIT_SIZE", "802"))
TEST_SPLIT_SIZE = int(os.environ.get("TEST_SPLIT_SIZE", "128"))
TRACKIO_SPACE = os.environ.get("TRACKIO_SPACE", "ZeterMordio/anchor-dashboard")
TRACKIO_PROJECT = os.environ.get("TRACKIO_PROJECT", "anchor-negotiation-pure")
RUN_NAME = os.environ.get("RUN_NAME", "")
PUSH_TRAINING_SCRIPT = os.environ.get("PUSH_TRAINING_SCRIPT", "1") == "1"

random.seed(SEED)
torch.manual_seed(SEED)


# ─── Dataset: AmazonHistoryPrice ─────────────────────────────────────────────
DATASET_URL_BASE = (
    "https://raw.githubusercontent.com/TianXiaSJTU/AmazonPriceHistory"
    "/main/data/AmazonHistoryPrice/"
)
# Important: include all 18 categories. Older scripts omitted toys-games and
# video-games, producing 901 examples. The paper/JOURNAL dataset has 930.
CATEGORIES = [
    "automotive",
    "baby-products",
    "beauty",
    "books",
    "electronics",
    "health-personal-care",
    "home-kitchen",
    "industrial-scientific",
    "movies-tv",
    "music",
    "other",
    "patio-lawn-garden",
    "pet-supplies",
    "software",
    "sports-outdoors",
    "tools-home-improvement",
    "toys-games",
    "video-games",
]


def parse_price(s):
    return float(str(s).replace("$", "").replace(",", "").strip())


def load_products(seed=SEED):
    import urllib.request

    all_items = []
    category_counts = {}
    for cat in CATEGORIES:
        url = DATASET_URL_BASE + f"{cat}.json"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                items = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  [WARN] Skip {cat}: {e}")
            continue
        n_valid = 0
        for idx, it in enumerate(items):
            try:
                lp = parse_price(it.get("list_price", "0"))
                cost = parse_price(it.get("lowest_price", "0"))
                if lp <= 0 or cost <= 0:
                    continue
                budget = round(lp * 0.8, 2)
                all_items.append(
                    {
                        "codename": f"{cat}_{idx}",
                        "title": it.get("title", "")[:120],
                        "description": it.get("description", "")[:200],
                        "features": it.get("features", "")[:300],
                        "current_price": parse_price(it.get("current_price", lp)),
                        "average_price": parse_price(it.get("average_price", lp)),
                        "highest_price": parse_price(it.get("highest_price", lp)),
                        "category": cat,
                        "list_price": lp,
                        "cost": cost,
                        "budget": budget,
                        "mi": budget > cost,
                    }
                )
                n_valid += 1
            except Exception:
                continue
        category_counts[cat] = n_valid

    random.seed(seed)
    random.shuffle(all_items)
    if len(all_items) >= TRAIN_SPLIT_SIZE + TEST_SPLIT_SIZE:
        train = all_items[:TRAIN_SPLIT_SIZE]
        test = all_items[TRAIN_SPLIT_SIZE : TRAIN_SPLIT_SIZE + TEST_SPLIT_SIZE]
    else:
        split = int(len(all_items) * 0.8623655913978494)  # 802/930
        train, test = all_items[:split], all_items[split:]

    mi_total = sum(1 for p in all_items if p["mi"])
    mi_train = sum(1 for p in train if p["mi"])
    mi_test = sum(1 for p in test if p["mi"])
    print(
        f"[DATA] Products={len(all_items)} train={len(train)} test={len(test)} "
        f"MI={mi_total} CI={len(all_items)-mi_total}",
        flush=True,
    )
    print(f"[DATA] Train MI={mi_train} CI={len(train)-mi_train}; Test MI={mi_test} CI={len(test)-mi_test}")
    print(f"[DATA] Category counts: {category_counts}")
    return train, test


# ─── Prompts (paper Appendix C, adapted to Thought/Talk/Action format) ───────
BUYER_SYSTEM = """You are a buyer looking forward to buying things on your Shopping List from me, the seller.
You have access to the seller's Inventory List and you can bargain about the prices.
Your task is to bargain with the seller and reach a deal with the price as low as possible in limited turns.
You can only buy things on the Shopping List in the limited quantity. Use the codename of the product, instead of the title.
You can only buy things that cost less than your budget, otherwise, you should quit negotiating.
Again, try to make deal with a price as low as possible. That is, your goal is to spend as little money as possible, not just reaching your budget.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the seller. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of "[BUY],[REJECT],[DEAL],[QUIT]".
1. '[BUY] $M (N codename_1)' if you wish to offer the seller $M to purchase all N items of the product with the codename "codename_1".
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the seller.
3. '[DEAL] $M (N codename_1)' if you finally accept on a former offer proposed by the seller. $M (N codename_1) is an exact copy of seller's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
You shouldn't choose action '[DEAL] $M' before seller's action '[SELL] $M'. Your first action should be '[BUY] $M (N codename_1)' or '[REJECT]'.
'[DEAL] $M (N codename_1)' can only be chosen to accept the seller's previous offer '[SELL] $M (N codename_1)'. Otherwise, you always choose from '[BUY]', '[REJECT]' and '[QUIT]'.

Your reply should STRICTLY follow this format (not following the format will directly lead to failure), for example:
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
Talk: short talk that you are going to say to the buyer. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation;
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
        f"  features: {product.get('features', '')}\n"
        f"  category: {product['category']}\n"
        f"  list_price: ${product['list_price']:.2f}\n"
        f"  current_price: ${product.get('current_price', product['list_price']):.2f}\n"
        f"  average_price: ${product.get('average_price', product['list_price']):.2f}\n"
        f"  highest_price: ${product.get('highest_price', product['list_price']):.2f}"
    )
    need = (
        f"Shopping List\n"
        f"- codename: {product['codename']}\n"
        f"  quantity: 1\n"
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
    inv = (
        f"Inventory List\n"
        f"- codename: {product['codename']}\n"
        f"  title: {product['title']}\n"
        f"  description: {product['description']}\n"
        f"  features: {product.get('features', '')}\n"
        f"  category: {product['category']}\n"
        f"  list_price: ${product['list_price']:.2f}\n"
        f"  current_price: ${product.get('current_price', product['list_price']):.2f}\n"
        f"  average_price: ${product.get('average_price', product['list_price']):.2f}\n"
        f"  highest_price: ${product.get('highest_price', product['list_price']):.2f}\n"
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


# ─── Action extraction + hidden scratchpad stripping ─────────────────────────
ACTION_PATTERN = r"\[(BUY|SELL|DEAL|REJECT|QUIT)\](?:\s*\$([\d,\.]+))?(?:\s*\(([^)]*)\))?"
ACTION_RE = re.compile(ACTION_PATTERN, re.IGNORECASE)
ACTION_LINE_RE = re.compile(r"(?:^|\n)\s*Action\s*:\s*" + ACTION_PATTERN, re.IGNORECASE)
QWEN_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
QWEN_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
QWEN_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def strip_qwen_native_thinking(text):
    """Remove Qwen3 native <think>...</think> content from visible text."""
    text = text or ""
    text = QWEN_THINK_BLOCK_RE.sub("", text)
    closes = list(QWEN_THINK_CLOSE_RE.finditer(text))
    if closes and not QWEN_THINK_OPEN_RE.search(text):
        text = text[closes[-1].end() :]
    m = QWEN_THINK_OPEN_RE.search(text)
    if m:
        tail = text[m.end() :]
        public = re.search(r"(?:^|\n)\s*(?:Thought|Talk|Action)\s*:", tail, re.IGNORECASE)
        text = text[: m.start()] + (tail[public.start() :] if public else "")
    return text.strip()


def _parse_action_match(m):
    ps = m.group(2)
    price = float(ps.replace(",", "")) if ps else None
    return {"type": m.group(1).upper(), "price": price, "objects": m.group(3)}


def extract_action(text):
    public_text = strip_qwen_native_thinking(text or "")
    line_matches = list(ACTION_LINE_RE.finditer(public_text))
    if line_matches:
        return _parse_action_match(line_matches[-1])
    public_text = strip_thought(public_text)
    matches = list(ACTION_RE.finditer(public_text or ""))
    m = matches[-1] if matches else None
    if m:
        return _parse_action_match(m)
    return {"type": "UNKNOWN", "price": None, "objects": None}


def replace_final_action(text, action_type, price, product):
    """Replace the final public structured action after environment regulation."""
    replacement = f"[{action_type}] ${price:.2f} (1x {product['codename']})"
    text = strip_qwen_native_thinking(text or "")
    line_matches = list(ACTION_LINE_RE.finditer(text))
    if line_matches:
        m = line_matches[-1]
        old_match = list(ACTION_RE.finditer(m.group(0)))[-1]
        start = m.start() + old_match.start()
        end = m.start() + old_match.end()
        return text[:start] + replacement + text[end:]
    visible = strip_thought(text)
    matches = list(ACTION_RE.finditer(visible))
    if not matches:
        return text.rstrip() + f"\nAction: {replacement}"
    old = matches[-1].group(0)
    idx = text.rfind(old)
    if idx < 0:
        return text.rstrip() + f"\nAction: {replacement}"
    return text[:idx] + replacement + text[idx + len(old) :]


def strip_thought(text):
    """Remove hidden scratchpads, keeping only Talk + Action for the counterparty.

    Paper §3.1: reasoning is a hidden scratchpad and is trimmed before being
    passed to the opponent. This also strips Qwen3 native <think> blocks so a
    Qwen thinking-mode ablation does not leak private reasoning.
    """
    text = strip_qwen_native_thinking(text or "")
    m = re.search(r"(?:^|\n)\s*Talk\s*:", text, re.IGNORECASE)
    if m:
        result = text[m.start() :].strip()
        _assert_strip_thought_complete(result, text)
        return result
    m = re.search(r"(?:^|\n)\s*Action\s*:", text, re.IGNORECASE)
    if m and re.search(r"(?:^|\n)\s*Thought\s*:", text[: m.start()], re.IGNORECASE):
        result = text[m.start() :].strip()
        _assert_strip_thought_complete(result, text)
        return result
    m = re.search(r"(?:^|\n)\s*Thought\s*:.*?(?=\n\s*(?:Talk|Action)\s*:)", text, re.IGNORECASE | re.DOTALL)
    if m:
        result = text[m.end() :].strip()
        _assert_strip_thought_complete(result, text)
        return result
    if re.search(r"(?:^|\n)\s*Thought\s*:", text, re.IGNORECASE):
        return ""
    _assert_strip_thought_complete(text, text)
    return text


def _assert_strip_thought_complete(stripped_text, original_text):
    has_structured_thought = bool(re.search(r"(?:^|\n)\s*Thought\s*:", original_text or ""))
    leaked_thought = bool(re.search(r"(?:^|\n)\s*Thought\s*:", stripped_text or ""))
    leaked_qwen_think = bool(QWEN_THINK_OPEN_RE.search(stripped_text or "") or QWEN_THINK_CLOSE_RE.search(stripped_text or ""))
    if (has_structured_thought and leaked_thought) or leaked_qwen_think:
        raise AssertionError(
            "strip_thought() INCOMPLETE: hidden reasoning block survived. "
            f"Original={original_text[:200]!r}; Stripped={stripped_text[:200]!r}"
        )


def _assert_no_private_info_leak(prompt_text, product, role):
    """Crash on clear counterparty-private-info leakage.

    Avoids regex heuristics that caused false positives in JOURNAL v10.4.
    """
    budget_str = f"${product['budget']:.2f}"
    if role == "buyer":
        if "cost_price" in prompt_text:
            raise AssertionError(
                f"INFORMATION LEAK: buyer prompt contains seller cost field. Product={product['codename']}"
            )
    elif role == "seller":
        if "Shopping List" in prompt_text:
            raise AssertionError(
                f"INFORMATION LEAK: seller prompt contains buyer Shopping List. Product={product['codename']}"
            )
        if f"budget_limit: {budget_str}" in prompt_text:
            raise AssertionError(
                f"INFORMATION LEAK: seller prompt contains buyer budget_limit={budget_str}. "
                f"Product={product['codename']}"
            )


# ─── Reward + seller regulation ──────────────────────────────────────────────
def regulate_seller(seller_action, buyer_price, product):
    """Regulate seller per paper: prevent below-cost accepts/proposals."""
    cost = product["cost"]
    at = seller_action["type"]
    price = seller_action["price"]

    if at == "UNKNOWN":
        return None, True, "SELLER_FORMAT_ERROR"
    if at == "QUIT":
        return None, True, "SELLER_QUIT"
    if at == "DEAL":
        if buyer_price is None:
            return None, True, "NO_PRIOR_BUYER_OFFER"
        if buyer_price < cost:
            return None, True, "SELLER_CANNOT_ACCEPT_BELOW_COST"
        return buyer_price, True, "DEAL_SELLER_ACCEPTS"
    if at == "SELL":
        if price is None:
            return None, True, "NO_PRICE_IN_SELL"
        if price < cost:
            price = round(cost * 1.05, 2)
        return price, False, "SELL"
    if at == "REJECT":
        return None, False, "REJECT"
    return None, True, f"UNEXPECTED_{at}"


def compute_buyer_reward(final_price, budget, cost, outcome):
    """Buyer reward per paper Eq. 1, with terminal penalties."""
    if "FORMAT_ERROR" in outcome or "UNEXPECTED" in outcome:
        return -1.0
    if outcome in {"BUYER_BUDGET_VIOLATION", "BUYER_DEAL_INVALID_SELLER_OFFER", "BUYER_DEAL_PRICE_MISMATCH"}:
        return -1.0
    if final_price is None:
        return 0.0
    if final_price > budget:
        return -1.0
    denom = abs(budget - cost)
    if denom < 1e-6:
        return 0.0
    r = (budget - final_price) / denom
    return max(-1.0, min(1.0, r))


# ─── Batched generation ──────────────────────────────────────────────────────
@torch.no_grad()
def generate_batched(model, tokenizer, prompts_text_list, max_new, temp, device):
    """Generate completions for a list of prompts using sub-batched HF generate."""
    if not prompts_text_list:
        return []

    all_results = []
    for batch_start in range(0, len(prompts_text_list), GEN_BATCH_LIMIT):
        batch_prompts = prompts_text_list[batch_start : batch_start + GEN_BATCH_LIMIT]
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
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
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(len(batch_prompts)):
            gen_tokens = output_ids[i][prompt_len:]
            gen_tokens = gen_tokens[gen_tokens != tokenizer.pad_token_id]
            # Protocol is Thought/Talk/Action with Qwen native thinking disabled.
            # If a backend/model still emits <think>, strip it before storing history;
            # keep our explicit Thought: field intact.
            all_results.append(strip_qwen_native_thinking(tokenizer.decode(gen_tokens, skip_special_tokens=True)))
    return all_results


# ─── Episode state/data ──────────────────────────────────────────────────────
@dataclass
class EpisodeState:
    product: dict
    idx: int
    buyer_texts: List[str] = field(default_factory=list)
    seller_texts: List[str] = field(default_factory=list)
    all_turns: List[Tuple[str, str]] = field(default_factory=list)
    final_price: Optional[float] = None
    outcome: str = "TIMEOUT"
    done: bool = False
    last_buyer_price: Optional[float] = None


@dataclass
class Episode:
    product: dict
    turns: List[Tuple[str, str]]  # [("buyer"|"seller", text), ...]
    final_price: Optional[float]
    reward: float
    num_turns: int
    outcome: str
    first_offer_price: Optional[float]
    budget_violations: int


def run_episodes_batched(buyer_model, seller_model, tokenizer, products_expanded, device):
    """Run buyer-only negotiation episodes with frozen seller, batched per turn."""
    states = [EpisodeState(product=p, idx=i) for i, p in enumerate(products_expanded)]

    for turn_round in range(MAX_TURNS):
        active_buyer = [s for s in states if not s.done]
        if not active_buyer:
            break

        buyer_prompts = []
        for s in active_buyer:
            msgs = build_buyer_prompt(s.product)
            for bt, st in zip(s.buyer_texts, s.seller_texts):
                msgs.append({"role": "assistant", "content": bt})  # own Thought kept
                msgs.append({"role": "user", "content": strip_thought(st)})
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            _assert_no_private_info_leak(prompt_text, s.product, "buyer")
            buyer_prompts.append(prompt_text)

        buyer_texts = generate_batched(buyer_model, tokenizer, buyer_prompts, MAX_NEW_TOKENS, BUYER_TEMP, device)

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
                    if last_s_act["type"] != "SELL" or last_s_act.get("price") is None:
                        s.outcome = "BUYER_DEAL_INVALID_SELLER_OFFER"
                        s.done = True
                    elif b_act.get("price") is not None and abs(b_act["price"] - last_s_act["price"]) > 0.01:
                        s.outcome = "BUYER_DEAL_PRICE_MISMATCH"
                        s.done = True
                    else:
                        s.final_price = last_s_act["price"]
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
                s.last_buyer_price = None
                still_active_for_seller.append(s)
            else:
                s.outcome = f"UNEXPECTED_{b_act['type']}"
                s.done = True

        if not still_active_for_seller:
            continue

        seller_prompts = []
        for s in still_active_for_seller:
            msgs = build_seller_prompt(s.product)
            for bt, st in zip(s.buyer_texts, s.seller_texts):
                msgs.append({"role": "user", "content": strip_thought(bt)})
                msgs.append({"role": "assistant", "content": st})  # seller own Thought kept
            if len(s.buyer_texts) > len(s.seller_texts):
                msgs.append({"role": "user", "content": strip_thought(s.buyer_texts[-1])})
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            _assert_no_private_info_leak(prompt_text, s.product, "seller")
            seller_prompts.append(prompt_text)

        seller_texts = generate_batched(seller_model, tokenizer, seller_prompts, MAX_NEW_TOKENS, SELLER_TEMP, device)

        for s, s_text in zip(still_active_for_seller, seller_texts):
            s_act = extract_action(s_text)
            r_price, done, reason = regulate_seller(s_act, s.last_buyer_price, s.product)
            if reason == "SELL" and r_price is not None and s_act.get("price") != r_price:
                # The regulated seller environment intercepts below-cost proposals.
                # Update the visible Action so future buyer context and reward parsing
                # use the valid regulated price, not the hallucinated below-cost one.
                s_text = replace_final_action(s_text, "SELL", r_price, s.product)
                s_act = extract_action(s_text)

            s.seller_texts.append(s_text)
            s.all_turns.append(("seller", s_text))

            if done:
                if reason == "DEAL_SELLER_ACCEPTS":
                    s.final_price = r_price
                s.outcome = reason
                s.done = True

    episodes = []
    for s in states:
        first_offer = None
        budget_violations = 0
        for role, text in s.all_turns:
            if role != "buyer":
                continue
            act = extract_action(text)
            if act["type"] == "BUY" and act["price"] is not None:
                if first_offer is None:
                    first_offer = act["price"]
                if act["price"] > s.product["budget"]:
                    budget_violations += 1
        reward = compute_buyer_reward(s.final_price, s.product["budget"], s.product["cost"], s.outcome)
        episodes.append(
            Episode(
                product=s.product,
                turns=s.all_turns,
                final_price=s.final_price,
                reward=reward,
                num_turns=len(s.all_turns),
                outcome=s.outcome,
                first_offer_price=first_offer,
                budget_violations=budget_violations,
            )
        )
    return episodes


# ─── Log-probs and pure buyer GRPO update ────────────────────────────────────
def _token_logprobs(model, input_ids, attention_mask):
    """Per-token log-probs using gather + logsumexp, avoiding full softmax tensor."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    target = input_ids[:, 1:].unsqueeze(-1)
    target_logit = torch.gather(logits, 2, target).squeeze(-1)
    log_z = torch.logsumexp(logits, dim=-1)
    return target_logit - log_z


def _norm_advantages(t):
    if t.numel() < 2 or not NORMALIZE_ADVANTAGES:
        return t
    return (t - t.mean()) / (t.std() + 1e-8)


def build_buyer_turn_prompt(ep, turn_idx):
    """Reconstruct buyer prompt for ep.turns[turn_idx]."""
    role, _ = ep.turns[turn_idx]
    assert role == "buyer"
    prompt_msgs = build_buyer_prompt(ep.product)
    for j in range(turn_idx):
        prev_role, prev_text = ep.turns[j]
        if prev_role == "buyer":
            prompt_msgs.append({"role": "assistant", "content": prev_text})
        else:
            prompt_msgs.append({"role": "user", "content": strip_thought(prev_text)})
    return prompt_msgs


def pure_grpo_update(buyer_model, ref_model, tokenizer, episodes, optimizer, device):
    """GRPO update on buyer turns only, grouped by product rollouts."""
    buyer_model.train()
    G = GROUP_SIZE
    num_groups = len(episodes) // G
    total_loss = 0.0
    turn_count = 0

    for g in range(num_groups):
        group_eps = episodes[g * G : (g + 1) * G]
        rewards = torch.tensor([ep.reward for ep in group_eps], dtype=torch.float32, device=device)
        advantages = _norm_advantages(rewards - rewards.mean())

        for _inner in range(NUM_INNER_EPOCHS):
            for i, ep in enumerate(group_eps):
                for turn_idx, (role, text) in enumerate(ep.turns):
                    if role != "buyer":
                        continue

                    prompt_msgs = build_buyer_turn_prompt(ep, turn_idx)
                    prompt_text = tokenizer.apply_chat_template(
                        prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    _assert_no_private_info_leak(prompt_text, ep.product, "buyer")
                    full_text = prompt_text + text

                    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)[
                        "input_ids"
                    ]
                    prompt_len = prompt_ids.shape[1]
                    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
                    ids = full_enc["input_ids"]
                    attn = full_enc["attention_mask"]

                    pol_lp = _token_logprobs(buyer_model, ids, attn)
                    with torch.no_grad():
                        ref_lp = _token_logprobs(ref_model, ids, attn)

                    mask = attn[:, 1:].clone()
                    mask[:, : prompt_len - 1] = 0

                    adv = advantages[i]
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

            torch.nn.utils.clip_grad_norm_(buyer_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(turn_count, 1)


# ─── Metrics / checkpoint helpers ────────────────────────────────────────────
def compute_iter_metrics(episodes):
    rewards = [ep.reward for ep in episodes]
    mean_r = sum(rewards) / max(len(rewards), 1)
    deals = [ep for ep in episodes if ep.final_price is not None]
    deal_rate = len(deals) / max(len(episodes), 1)
    mean_price = sum(ep.final_price for ep in deals) / max(len(deals), 1) if deals else 0.0
    mean_turns = sum(ep.num_turns for ep in episodes) / max(len(episodes), 1)
    outcomes: Dict[str, int] = {}
    for ep in episodes:
        outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1
    role_confusions = 0
    for ep in episodes:
        for role, text in ep.turns:
            act = extract_action(text)
            if role == "buyer" and act["type"] == "SELL":
                role_confusions += 1
            elif role == "seller" and act["type"] == "BUY":
                role_confusions += 1
    budget_violations = sum(1 for ep in episodes if ep.budget_violations > 0)
    first_offer_ratios = [ep.first_offer_price / ep.product["budget"] for ep in episodes if ep.first_offer_price]
    mean_first_offer_ratio = sum(first_offer_ratios) / max(len(first_offer_ratios), 1) if first_offer_ratios else None

    return {
        "mean_reward": mean_r,
        "deal_rate": deal_rate,
        "mean_price": mean_price,
        "mean_turns": mean_turns,
        "outcomes": dict(sorted(outcomes.items(), key=lambda x: -x[1])),
        "role_confusions": role_confusions,
        "price_overshoot_rate": budget_violations / max(len(episodes), 1),
        "first_offer_ratio": mean_first_offer_ratio,
    }


def save_and_push_checkpoint(buyer_model, tokenizer, metrics, iteration, final=False):
    if not HUB_MODEL_ID:
        return
    branch = "main" if final else f"iter-{iteration + 1}"
    label = "FINAL" if final else f"iter {iteration + 1}"
    path = Path(OUTPUT_DIR if final else f"/tmp/pure-ckpt-{iteration+1}")
    path.mkdir(parents=True, exist_ok=True)
    buyer_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if PUSH_TRAINING_SCRIPT:
        try:
            shutil.copyfile(__file__, path / "train_negotiation_pure.py")
        except Exception:
            pass

    try:
        from huggingface_hub import HfApi, create_repo

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        create_repo(HUB_MODEL_ID, exist_ok=True, token=token)
        if not final:
            try:
                api.create_branch(HUB_MODEL_ID, branch=branch, repo_type="model")
            except Exception:
                pass
        api.upload_folder(
            folder_path=path,
            repo_id=HUB_MODEL_ID,
            repo_type="model",
            revision=branch,
            commit_message=f"Pure negotiation {label}",
        )
        print(f"  [CHECKPOINT] ✅ Pushed {label} to {HUB_MODEL_ID}@{branch}")
    except Exception as e:
        print(f"  [CHECKPOINT] ⚠️ Push failed (non-fatal): {e}")
    finally:
        if not final:
            shutil.rmtree(path, ignore_errors=True)


# ─── Main ────────────────────────────────────────────────────────────────────
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
    print("=" * 70, flush=True)


def main():
    check_cuda()

    print(f"[CONFIG] Pure Negotiation-RLVR buyer-only training")
    print(f"[CONFIG] BuyerModel={MODEL_NAME} SellerModel={SELLER_MODEL_NAME}")
    print(f"[CONFIG] Iters={NUM_ITERS} Batch={BATCH_SIZE} Group={GROUP_SIZE} Episodes/iter={BATCH_SIZE*GROUP_SIZE}")
    print(f"[CONFIG] Turns={MAX_TURNS} LR={LR} Eps={EPSILON} KL={KL_COEF}")
    print(f"[CONFIG] BuyerTemp={BUYER_TEMP} SellerTemp={SELLER_TEMP} MaxNew={MAX_NEW_TOKENS}")
    print(f"[CONFIG] GradCheckpoint={GRADIENT_CHECKPOINTING} GenBatchLimit={GEN_BATCH_LIMIT}")
    print(f"[CONFIG] InnerEpochs={NUM_INNER_EPOCHS} NormAdvantages={NORMALIZE_ADVANTAGES}")
    print(f"[CONFIG] CheckpointEvery={CHECKPOINT_EVERY} Hub={HUB_MODEL_ID or '(disabled)'}")
    print("=" * 70, flush=True)

    try:
        import trackio

        run_name = RUN_NAME or f"pure-{MODEL_NAME.split('/')[-1]}-{NUM_ITERS}it"
        trackio.init(
            project=TRACKIO_PROJECT,
            name=run_name,
            space_id=TRACKIO_SPACE,
            config={
                "method": "pure_negotiation_rlvr",
                "buyer_model": MODEL_NAME,
                "seller_model": SELLER_MODEL_NAME,
                "num_iters": NUM_ITERS,
                "batch_size": BATCH_SIZE,
                "group_size": GROUP_SIZE,
                "max_turns": MAX_TURNS,
                "lr": LR,
                "epsilon": EPSILON,
                "kl_coef": KL_COEF,
                "max_new_tokens": MAX_NEW_TOKENS,
                "buyer_temp": BUYER_TEMP,
                "seller_temp": SELLER_TEMP,
                "normalize_advantages": NORMALIZE_ADVANTAGES,
                "num_inner_epochs": NUM_INNER_EPOCHS,
                "liger_kernel": USE_LIGER,
                "dataset_categories": CATEGORIES,
            },
        )
        TRACKIO_OK = True
        print(f"[TRACKIO] Dashboard: https://huggingface.co/spaces/{TRACKIO_SPACE}")
    except Exception as e:
        print(f"[TRACKIO] Init failed (non-fatal): {e}")
        TRACKIO_OK = False
        trackio = None

    print("\n[1/5] Loading dataset...")
    train_products, _ = load_products(seed=SEED)

    print(f"\n[2/5] Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  [OK]")

    print(f"\n[3/5] Loading trainable buyer model...")
    buyer_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if GRADIENT_CHECKPOINTING:
        buyer_model.gradient_checkpointing_enable()
        if hasattr(buyer_model, "config"):
            buyer_model.config.use_cache = False
    dev = next(buyer_model.parameters()).device
    print(f"  [OK] Device={dev} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    print(f"\n[4/5] Loading frozen seller/reference model...")
    seller_model = AutoModelForCausalLM.from_pretrained(
        SELLER_MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    seller_model.eval()
    for p in seller_model.parameters():
        p.requires_grad = False
    ref_model = seller_model if SELLER_MODEL_NAME == MODEL_NAME else AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if ref_model is not seller_model:
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    print(f"  [OK] VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    print(f"\n[5/5] Optimizer (AdamW, lr={LR}, betas=(0.9,0.95), wd=0.0)...")
    optimizer = torch.optim.AdamW(buyer_model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    n_params = sum(p.numel() for p in buyer_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    print(f"\n{'=' * 70}\nPURE NEGOTIATION-RLVR TRAINING\n{'=' * 70}")
    metrics = []
    t0 = time.time()

    for iteration in range(NUM_ITERS):
        t_iter = time.time()
        print(f"\n--- Iteration {iteration} ---")
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

        products = random.sample(train_products, min(BATCH_SIZE, len(train_products)))
        products_expanded = [p for p in products for _ in range(GROUP_SIZE)]
        n_episodes = len(products_expanded)
        print(f"  Sampling {len(products)} products × {GROUP_SIZE} rollouts = {n_episodes} episodes...")

        buyer_model.eval()
        seller_model.eval()
        rollout_t0 = time.time()
        episodes = run_episodes_batched(buyer_model, seller_model, tokenizer, products_expanded, dev)
        rollout_time = time.time() - rollout_t0
        print(f"  Rollout: {n_episodes} episodes in {rollout_time:.0f}s ({rollout_time/n_episodes:.1f}s/ep)")
        torch.cuda.empty_cache()
        gc.collect()

        buyer_model.train()
        print(f"  Pure GRPO update on buyer turns only...")
        loss = pure_grpo_update(buyer_model, ref_model, tokenizer, episodes, optimizer, dev)
        torch.cuda.empty_cache()
        gc.collect()

        iter_metrics = compute_iter_metrics(episodes)
        elapsed = time.time() - t_iter
        update_time = elapsed - rollout_time
        current_vram = torch.cuda.memory_allocated() / 1e9
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        print(
            f"  Loss={loss:.4f} Reward={iter_metrics['mean_reward']:.4f} "
            f"Deal={iter_metrics['deal_rate']:.1%} Price=${iter_metrics['mean_price']:.2f} "
            f"Turns={iter_metrics['mean_turns']:.1f}"
        )
        print(f"  FirstOfferRatio={iter_metrics['first_offer_ratio']} Overshoot={iter_metrics['price_overshoot_rate']:.1%}")
        print(f"  Time={elapsed:.0f}s (rollout={rollout_time:.0f}s update={update_time:.0f}s)")
        print(f"  Outcomes: {dict(list(iter_metrics['outcomes'].items())[:6])}")
        if iter_metrics["role_confusions"]:
            print(f"  ⚠️ ROLE CONFUSIONS: {iter_metrics['role_confusions']}")
        print(f"  VRAM: {current_vram:.1f}GB current, {peak_vram:.1f}GB peak", flush=True)

        row = {
            "iteration": iteration,
            "loss": loss,
            **iter_metrics,
            "time": elapsed,
            "rollout_time": rollout_time,
            "update_time": update_time,
            "vram_current_gb": current_vram,
            "vram_peak_gb": peak_vram,
        }
        metrics.append(row)

        if TRACKIO_OK:
            try:
                trackio.log(
                    {
                        "train/loss": loss,
                        "reward/buyer": iter_metrics["mean_reward"],
                        "negotiation/deal_rate": iter_metrics["deal_rate"],
                        "negotiation/mean_price": iter_metrics["mean_price"],
                        "negotiation/mean_turns": iter_metrics["mean_turns"],
                        "negotiation/first_offer_ratio": iter_metrics["first_offer_ratio"] or 0.0,
                        "negotiation/price_overshoot_rate": iter_metrics["price_overshoot_rate"],
                        "perf/iter_time_s": elapsed,
                        "perf/rollout_time_s": rollout_time,
                        "perf/update_time_s": update_time,
                        "perf/vram_gb": current_vram,
                        "perf/vram_peak_gb": peak_vram,
                        "sanity/role_confusions": iter_metrics["role_confusions"],
                    },
                    step=iteration,
                )
                if iteration == 0:
                    trackio.alert(
                        "pure_negotiation_started",
                        f"iter=0 reward={iter_metrics['mean_reward']:.4f} deal_rate={iter_metrics['deal_rate']:.3f}; continue 42-iter run if format errors stay low",
                        level=trackio.AlertLevel.INFO,
                    )
                if iter_metrics["mean_reward"] < -0.5:
                    trackio.alert(
                        "low_reward_warning",
                        f"reward={iter_metrics['mean_reward']:.4f} at iter={iteration}; if persistent, reduce LR or increase KL anchor",
                        level=trackio.AlertLevel.WARN,
                    )
                fmt_errors = iter_metrics["outcomes"].get("BUYER_FORMAT_ERROR", 0)
                if fmt_errors > 0.25 * n_episodes:
                    trackio.alert(
                        "format_collapse_warning",
                        f"buyer_format_errors={fmt_errors}/{n_episodes} at iter={iteration}; try LR x0.1 or KL x2",
                        level=trackio.AlertLevel.WARN,
                    )
            except Exception as e:
                print(f"  [TRACKIO] Log/alert failed (non-fatal): {e}")

        should_ckpt = (
            CHECKPOINT_EVERY > 0
            and (iteration + 1) % CHECKPOINT_EVERY == 0
            and iteration < NUM_ITERS - 1
        )
        if should_ckpt:
            print(f"  [CHECKPOINT] Saving iter {iteration+1}...")
            save_and_push_checkpoint(buyer_model, tokenizer, metrics, iteration, final=False)

    print(f"\n{'=' * 70}\nSAVING FINAL\n{'=' * 70}")
    save_path = Path(OUTPUT_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    buyer_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if PUSH_TRAINING_SCRIPT:
        try:
            shutil.copyfile(__file__, save_path / "train_negotiation_pure.py")
        except Exception:
            pass
    print(f"  Saved to {save_path}")

    if HUB_MODEL_ID:
        save_and_push_checkpoint(buyer_model, tokenizer, metrics, NUM_ITERS - 1, final=True)

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"COMPLETE  Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"{'=' * 70}")

    if TRACKIO_OK:
        try:
            trackio.alert("pure_negotiation_complete", f"iters={NUM_ITERS}; final_reward={metrics[-1]['mean_reward']:.4f}; final_deal_rate={metrics[-1]['deal_rate']:.3f}", level=trackio.AlertLevel.INFO)
            trackio.finish()
            print(f"[TRACKIO] Finished. Dashboard: https://huggingface.co/spaces/{TRACKIO_SPACE}")
        except Exception as e:
            print(f"[TRACKIO] Finish failed (non-fatal): {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
