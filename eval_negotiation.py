"""
eval_negotiation.py — Comprehensive evaluation for trained negotiation models.

Implements the RLVR paper (2604.09855) §4.1 metrics + §5 benchmarking protocol:
  - Reward R ∈ [-1, 1]: primary optimization target (surplus + constraint satisfaction)
  - Deal Rate: fraction of episodes concluding with an agreement
  - Bargained Ratio: surplus extraction on SUCCESSFUL DEALS ONLY (B-P)/(B-C)
  - First-Turn Offer Ratio: Offer_1 / Budget (measures anchoring aggression)
  - Price Overshoot Rate: frequency of budget violations (buyer proposes > budget)
  - MI/CI split: separate metrics for Mutual Interest vs Conflict of Interest scenarios

Paper evaluation protocol (Table 6, Appendix A.2):
  - 128 test products × 4 rollouts = 512 episodes
  - buyer_temperature = 1.0, seller_temperature = 0.7
  - max_tokens = 4000 (evaluation), max_turns = 6
  - Seller uses neutral persona (no adversarial traits)

Key design decisions for the current SDPO-first setup:
  - Buyer = trained checkpoint (e.g. iter-10, iter-20, ... or final main)
  - Seller = frozen base model from the same Qwen family when possible
  - This gives apples-to-apples comparison: same regulated seller across all checkpoints
  - Hidden Thought / native <think> blocks are stripped before opponent visibility
  - Seller regulation: cannot accept below cost (same as training)

Usage:
    # Evaluate current SDPO main model against frozen base seller
    BUYER_PATH=ZeterMordio/anchor-negotiation-sdpo \
    SELLER_PATH=Qwen/Qwen3-8B \
    python eval_negotiation.py

    # Evaluate a specific checkpoint
    BUYER_PATH=ZeterMordio/anchor-negotiation-sdpo \
    BUYER_REV=iter-20 \
    SELLER_PATH=Qwen/Qwen3-8B \
    python eval_negotiation.py

    # Evaluate untrained baseline (buyer = seller = same base model)
    BUYER_PATH=Qwen/Qwen3-4B-Instruct-2507 \
    SELLER_PATH=Qwen/Qwen3-4B-Instruct-2507 \
    python eval_negotiation.py

    # Multi-checkpoint sweep (run from shell)
    for CKPT in iter-10 iter-20 iter-30 iter-40 iter-50 main; do
      BUYER_REV=$CKPT OUT=eval_${CKPT}.json python eval_negotiation.py
    done
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
from collections import defaultdict

os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # Older Transformers cannot load Qwen3.5 image-text checkpoints.
    AutoModelForImageTextToText = None

# ─── Config ────────────────────────────────────────────────────────────────
BUYER_PATH   = os.environ.get("BUYER_PATH", "ZeterMordio/anchor-negotiation-sdpo")
BUYER_REV    = os.environ.get("BUYER_REV", None)  # HF branch/tag, e.g. "iter-20"
SELLER_PATH  = os.environ.get("SELLER_PATH", "Qwen/Qwen3-8B")
N_TEST       = int(os.environ.get("N_TEST", "128"))   # paper: 128 test products
N_ROLL       = int(os.environ.get("N_ROLL", "4"))     # paper: 4 rollouts per product
MAX_TURNS    = int(os.environ.get("MAX_TURNS", "6"))   # paper: 6
BUYER_TEMP   = float(os.environ.get("BUYER_TEMP", "1.0"))   # paper: 1.0
SELLER_TEMP  = float(os.environ.get("SELLER_TEMP", "0.7"))  # paper: 0.7
MAX_NEW      = int(os.environ.get("MAX_NEW", "4000"))  # paper eval: 4000 (not 300 like training)
SEED         = int(os.environ.get("SEED", "42"))
TRAIN_SPLIT_SIZE = int(os.environ.get("TRAIN_SPLIT_SIZE", "802"))
TEST_SPLIT_SIZE  = int(os.environ.get("TEST_SPLIT_SIZE", "128"))
OUT          = os.environ.get("OUT", "eval_results.json")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")  # optional: push results to model repo
USE_LIGER    = os.environ.get("USE_LIGER", "1") == "1"
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
CHAT_TEMPLATE_ENABLE_THINKING = os.environ.get("CHAT_TEMPLATE_ENABLE_THINKING", "0") == "1"

random.seed(SEED)

_IMAGE_TEXT_MODEL_TYPES = {"qwen3_5", "qwen3_vl", "qwen2_5_vl", "qwen2_vl"}
_TEXT_BATCH_KEYS = {"input_ids", "attention_mask", "position_ids"}


# ─── Liger Kernel (optional, for faster inference) ──────────────────────────
if USE_LIGER:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3()
        print("[LIGER] Qwen3 kernels patched")
    except Exception:
        USE_LIGER = False

# ─── Dataset (identical to training — same seed, same split) ────────────────
DATASET_URL_BASE = (
    "https://raw.githubusercontent.com/TianXiaSJTU/AmazonPriceHistory"
    "/main/data/AmazonHistoryPrice/"
)
CATEGORIES = [
    "automotive", "baby-products", "beauty", "books", "electronics",
    "health-personal-care", "home-kitchen", "industrial-scientific",
    "movies-tv", "music", "other", "patio-lawn-garden", "pet-supplies",
    "software", "sports-outdoors", "tools-home-improvement", "toys-games",
    "video-games",
]

def parse_price(s):
    return float(str(s).replace("$", "").replace(",", "").strip())

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
                    "features": it.get("features", "")[:300],
                    "current_price": parse_price(it.get("current_price", lp)),
                    "average_price": parse_price(it.get("average_price", lp)),
                    "highest_price": parse_price(it.get("highest_price", lp)),
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
    if len(all_items) >= TRAIN_SPLIT_SIZE + TEST_SPLIT_SIZE:
        train = all_items[:TRAIN_SPLIT_SIZE]
        test = all_items[TRAIN_SPLIT_SIZE : TRAIN_SPLIT_SIZE + TEST_SPLIT_SIZE]
    else:
        split = int(len(all_items) * 0.8623655913978494)  # 802/930, training-script parity
        train, test = all_items[:split], all_items[split:]
    mi = sum(1 for p in test if p["mi"])
    print(f"[DATA] Total={len(all_items)} train={len(train)} test={len(test)} test_MI={mi} test_CI={len(test)-mi}")
    return train, test


# ─── Prompts (identical to training script) ──────────────────────────────────
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


# ─── Action extraction (identical to training) ──────────────────────────────
# Capture numeric prices without swallowing trailing sentence punctuation such as
# "$25.00."; the old [\d,.]+ pattern could include the final period and crash.
PRICE_PATTERN = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
ACTION_PATTERN = r"\[(BUY|SELL|DEAL|REJECT|QUIT)\](?:\s*\$" + PRICE_PATTERN + r")?(?:\s*\(([^)]*)\))?"
ACTION_RE = re.compile(ACTION_PATTERN, re.IGNORECASE)
ACTION_LINE_RE = re.compile(r'(?:^|\n)\s*Action\s*:\s*' + ACTION_PATTERN, re.IGNORECASE)
QWEN_THINK_BLOCK_RE = re.compile(r'<think\b[^>]*>.*?</think\s*>', re.IGNORECASE | re.DOTALL)
QWEN_THINK_OPEN_RE = re.compile(r'<think\b[^>]*>', re.IGNORECASE)
QWEN_THINK_CLOSE_RE = re.compile(r'</think\s*>', re.IGNORECASE)


def strip_qwen_native_thinking(text):
    """Remove Qwen3 native <think>...</think> content from visible text."""
    text = text or ""
    text = QWEN_THINK_BLOCK_RE.sub("", text)
    closes = list(QWEN_THINK_CLOSE_RE.finditer(text))
    if closes and not QWEN_THINK_OPEN_RE.search(text):
        text = text[closes[-1].end():]
    m = QWEN_THINK_OPEN_RE.search(text)
    if m:
        tail = text[m.end():]
        public = re.search(r'(?:^|\n)\s*(?:Thought|Talk|Action)\s*:', tail, re.IGNORECASE)
        text = text[:m.start()] + (tail[public.start():] if public else "")
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
    """Remove hidden scratchpads, keep only Talk + Action (for cross-role context)."""
    text = strip_qwen_native_thinking(text or "")
    m = re.search(r'(?:^|\n)\s*Talk\s*:', text, re.IGNORECASE)
    if m:
        return text[m.start():].strip()
    m = re.search(r'(?:^|\n)\s*Action\s*:', text, re.IGNORECASE)
    if m and re.search(r'(?:^|\n)\s*Thought\s*:', text[:m.start()], re.IGNORECASE):
        return text[m.start():].strip()
    m = re.search(r'(?:^|\n)\s*Thought\s*:.*?(?=\n\s*(?:Talk|Action)\s*:)', text, re.IGNORECASE | re.DOTALL)
    if m:
        return text[m.end():].strip()
    if re.search(r'(?:^|\n)\s*Thought\s*:', text, re.IGNORECASE):
        return ""
    return text


# ─── Seller regulation (identical to training) ──────────────────────────────
def regulate_seller(seller_action, buyer_price, product):
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
            price = cost * 1.05
        return price, False, "SELL"
    if at == "REJECT":
        return None, False, "REJECT"
    return None, True, f"UNEXPECTED_{at}"


# ─── Reward (identical to training) ──────────────────────────────────────────
def compute_buyer_reward(final_price, budget, cost, outcome):
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


# ─── Model loading / generation ──────────────────────────────────────────────
def _is_image_text_config(config):
    return getattr(config, "vision_config", None) is not None or str(getattr(config, "model_type", "")) in _IMAGE_TEXT_MODEL_TYPES


def _from_pretrained_compat(cls, model_id, **kwargs):
    """Use current dtype=... API, with torch_dtype fallback for older installs."""
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except TypeError as e:
        if "dtype" not in str(e) or "dtype" not in kwargs:
            raise
        kwargs = dict(kwargs)
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return cls.from_pretrained(model_id, **kwargs)


def _model_input_device(model):
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        for key in ("model.embed_tokens", "language_model.model.embed_tokens", "model", "language_model"):
            if key in hf_map:
                dev = torch.device(hf_map[key])
                if dev.type == "cpu":
                    raise RuntimeError(f"Input layer {key} is on CPU in hf_device_map; use larger GPU memory")
                return dev
        dev = torch.device(next(iter(hf_map.values())))
        if dev.type == "cpu":
            raise RuntimeError("First model shard is on CPU; use larger GPU memory")
        return dev
    return next(model.parameters()).device


def _as_vlm_messages(messages):
    out = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        out.append({"role": msg["role"], "content": content})
    return out


def _text_only_batch(batch):
    return {k: v for k, v in batch.items() if k in _TEXT_BATCH_KEYS and v is not None}


def _move_batch_to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def load_eval_stack(model_id, revision=None):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, **({"revision": revision} if revision else {}))
    common = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        **({"revision": revision} if revision else {}),
    }
    if ATTN_IMPLEMENTATION:
        common["attn_implementation"] = ATTN_IMPLEMENTATION

    if _is_image_text_config(cfg):
        if AutoModelForImageTextToText is None:
            raise RuntimeError(
                f"{model_id} looks like an image-text checkpoint but this Transformers install lacks "
                "AutoModelForImageTextToText. Upgrade transformers."
            )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **({"revision": revision} if revision else {}))
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"AutoProcessor for {model_id} does not expose .tokenizer")
        model = _from_pretrained_compat(AutoModelForImageTextToText, model_id, **common)
        stack = {"model": model, "processor": processor, "tokenizer": tokenizer, "is_image_text": True}
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **({"revision": revision} if revision else {}))
        model = _from_pretrained_compat(AutoModelForCausalLM, model_id, **common)
        stack = {"model": model, "processor": tokenizer, "tokenizer": tokenizer, "is_image_text": False}

    if stack["tokenizer"].pad_token is None:
        stack["tokenizer"].pad_token = stack["tokenizer"].eos_token
    model.eval()
    return stack


@torch.no_grad()
def generate_single(stack, messages, max_new, temp):
    """Generate a single response. Supports CausalLM and Qwen3.5 image-text wrappers."""
    model = stack["model"]
    tokenizer = stack["tokenizer"]
    device = _model_input_device(model)
    gen_kwargs = {
        "max_new_tokens": max_new,
        "do_sample": True,
        "temperature": max(temp, 0.01),
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if stack["is_image_text"]:
        processor = stack["processor"]
        inputs = processor.apply_chat_template(
            _as_vlm_messages(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=CHAT_TEMPLATE_ENABLE_THINKING,
        )
        inputs.pop("token_type_ids", None)
        inputs = _move_batch_to_device(_text_only_batch(inputs), device)
        output_ids = model.generate(**inputs, **gen_kwargs)
        gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = processor.batch_decode([gen_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=CHAT_TEMPLATE_ENABLE_THINKING,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = _move_batch_to_device(_text_only_batch(inputs), device)
        output_ids = model.generate(**inputs, **gen_kwargs)
        gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Evaluation defaults to no native thinking; if a backend/model still emits
    # <think>, strip it before storing/evaluating/opponent visibility.
    return strip_qwen_native_thinking(text)


# ─── Episode data ────────────────────────────────────────────────────────────
@dataclass
class EvalEpisode:
    product: dict
    buyer_texts: List[str] = field(default_factory=list)
    seller_texts: List[str] = field(default_factory=list)
    final_price: Optional[float] = None
    reward: float = 0.0
    num_turns: int = 0
    outcome: str = "TIMEOUT"
    first_offer_price: Optional[float] = None  # buyer's first [BUY] price
    all_buyer_prices: List[float] = field(default_factory=list)  # every [BUY] price proposed
    budget_violations: int = 0  # count of [BUY] $X where X > budget


def run_eval_episode(buyer_stack, seller_stack, product):
    """Run one evaluation episode: trained buyer vs frozen seller.
    
    Protocol matches the RLVR paper:
    - Buyer always goes first
    - Thought blocks stripped before passing to counterparty
    - Seller regulated (cannot accept below cost)
    - max_turns = 6 rounds
    """
    ep = EvalEpisode(product=product)
    
    buyer_msgs = build_buyer_prompt(product)
    seller_msgs = build_seller_prompt(product)
    last_buyer_price = None
    
    for turn in range(MAX_TURNS):
        # ── Buyer turn ──
        b_text = generate_single(buyer_stack, buyer_msgs, MAX_NEW, BUYER_TEMP)
        b_act = extract_action(b_text)
        ep.buyer_texts.append(b_text)
        ep.num_turns += 1
        
        # Track buyer prices
        if b_act["type"] == "BUY" and b_act["price"] is not None:
            ep.all_buyer_prices.append(b_act["price"])
            if ep.first_offer_price is None:
                ep.first_offer_price = b_act["price"]
            if b_act["price"] > product["budget"]:
                ep.budget_violations += 1
        
        # Add buyer's full text to buyer's own history (keeps Thought for chain-of-thought)
        buyer_msgs.append({"role": "assistant", "content": b_text})
        
        # Process buyer action
        if b_act["type"] == "QUIT":
            ep.outcome = "BUYER_QUIT"
            break
        elif b_act["type"] == "UNKNOWN":
            ep.outcome = "BUYER_FORMAT_ERROR"
            break
        elif b_act["type"] == "DEAL":
            if not ep.seller_texts:
                ep.outcome = "BUYER_DEAL_NO_SELLER_OFFER"
                break
            last_s_act = extract_action(ep.seller_texts[-1])
            if last_s_act["type"] != "SELL" or last_s_act.get("price") is None:
                ep.outcome = "BUYER_DEAL_INVALID_SELLER_OFFER"
                break
            if b_act.get("price") is not None and abs(b_act["price"] - last_s_act["price"]) > 0.01:
                ep.outcome = "BUYER_DEAL_PRICE_MISMATCH"
                break
            ep.final_price = last_s_act["price"]
            ep.outcome = "DEAL_BUYER_ACCEPTS"
            break
        elif b_act["type"] == "BUY":
            if b_act["price"] is not None and b_act["price"] > product["budget"]:
                ep.outcome = "BUYER_BUDGET_VIOLATION"
                break
            last_buyer_price = b_act["price"]
        elif b_act["type"] == "REJECT":
            last_buyer_price = None  # no price on reject
        else:
            ep.outcome = f"UNEXPECTED_{b_act['type']}"
            break
        
        # ── Seller turn ──
        # Add buyer's STRIPPED text to seller's context
        seller_msgs.append({"role": "user", "content": strip_thought(b_text)})
        
        s_text = generate_single(seller_stack, seller_msgs, MAX_NEW, SELLER_TEMP)
        s_act = extract_action(s_text)
        ep.seller_texts.append(s_text)
        
        # Regulate seller before storing cross-role context, matching training.
        r_price, done, reason = regulate_seller(s_act, last_buyer_price, product)
        if reason == "SELL" and r_price is not None and s_act.get("price") != r_price:
            s_text = replace_final_action(s_text, "SELL", r_price, product)
            s_act = extract_action(s_text)
            ep.seller_texts[-1] = s_text

        # Add seller's full text to seller's own history
        seller_msgs.append({"role": "assistant", "content": s_text})
        # Add seller's STRIPPED text to buyer's context
        buyer_msgs.append({"role": "user", "content": strip_thought(s_text)})
        
        if done:
            if "DEAL" in reason:
                ep.final_price = r_price
            ep.outcome = reason
            break
    
    # Compute reward
    ep.reward = compute_buyer_reward(ep.final_price, product["budget"], product["cost"], ep.outcome)
    return ep


# ─── Metrics computation (paper §4.1 + §5) ──────────────────────────────────
def compute_metrics(episodes: List[EvalEpisode], label: str = "all") -> Dict:
    """Compute all paper metrics from a list of episodes.
    
    Returns dict with:
      reward: mean R across all episodes (paper's primary metric)
      reward_std: standard deviation of R
      reward_se: standard error of mean R
      deal_rate: fraction of episodes with a deal
      bargained_ratio: mean (B-P)/(B-C) on SUCCESSFUL DEALS ONLY
      first_offer_ratio: mean Offer_1/Budget on episodes where buyer made a BUY
      price_overshoot_rate: fraction of episodes with ANY budget violation
      mean_turns: average episode length
      outcome_counts: dict of outcome → count
    """
    n = len(episodes)
    if n == 0:
        return {"label": label, "n": 0}
    
    rewards = [ep.reward for ep in episodes]
    mean_r = sum(rewards) / n
    
    import math
    variance = sum((r - mean_r) ** 2 for r in rewards) / max(n - 1, 1)
    std_r = math.sqrt(variance)
    se_r = std_r / math.sqrt(n)
    
    # Deal rate
    deals = [ep for ep in episodes if ep.final_price is not None]
    deal_rate = len(deals) / n
    
    # Bargained ratio: (budget - final_price) / (budget - cost), on DEALS ONLY
    # This is the paper's "Buyer Bargained Ratio" — measures surplus extraction efficiency
    bargained_ratios = []
    for ep in deals:
        denom = ep.product["budget"] - ep.product["cost"]
        if abs(denom) > 1e-6:
            br = (ep.product["budget"] - ep.final_price) / denom
            bargained_ratios.append(br)
    mean_bargained = sum(bargained_ratios) / max(len(bargained_ratios), 1) if bargained_ratios else 0.0
    
    # First-turn offer ratio: Offer_1 / Budget
    # Measures anchoring aggression. Near 1.0 = naive, low = aggressive
    offer_ratios = []
    for ep in episodes:
        if ep.first_offer_price is not None and ep.product["budget"] > 0:
            offer_ratios.append(ep.first_offer_price / ep.product["budget"])
    mean_offer_ratio = sum(offer_ratios) / max(len(offer_ratios), 1) if offer_ratios else None
    
    # Price overshoot rate: fraction of episodes where buyer EVER proposed > budget
    overshoots = sum(1 for ep in episodes if ep.budget_violations > 0)
    overshoot_rate = overshoots / n
    
    # Mean turns
    mean_turns = sum(ep.num_turns for ep in episodes) / n
    
    # Outcome distribution
    outcomes = defaultdict(int)
    for ep in episodes:
        outcomes[ep.outcome] += 1
    
    return {
        "label": label,
        "n": n,
        "reward": round(mean_r, 4),
        "reward_std": round(std_r, 4),
        "reward_se": round(se_r, 4),
        "deal_rate": round(deal_rate, 4),
        "bargained_ratio": round(mean_bargained, 4),
        "first_offer_ratio": round(mean_offer_ratio, 4) if mean_offer_ratio is not None else None,
        "price_overshoot_rate": round(overshoot_rate, 4),
        "mean_turns": round(mean_turns, 2),
        "outcomes": dict(sorted(outcomes.items(), key=lambda x: -x[1])),
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print(f"{'=' * 70}")
    print(f"NEGOTIATION EVALUATION — RLVR Paper Protocol")
    print(f"{'=' * 70}")
    print(f"Buyer:  {BUYER_PATH}" + (f" (rev: {BUYER_REV})" if BUYER_REV else ""))
    print(f"Seller: {SELLER_PATH}")
    print(f"Test:   {N_TEST} products × {N_ROLL} rollouts = {N_TEST * N_ROLL} episodes")
    print(f"Config: max_turns={MAX_TURNS} buyer_temp={BUYER_TEMP} seller_temp={SELLER_TEMP} max_new={MAX_NEW}")
    print(f"Split:  train={TRAIN_SPLIT_SIZE} test={TEST_SPLIT_SIZE} categories={len(CATEGORIES)}")
    print(f"Seed:   {SEED}")
    print(f"Output: {OUT}")
    print(f"{'=' * 70}\n")
    
    # CUDA check
    if not torch.cuda.is_available():
        print("FATAL: No CUDA")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")
    
    # Load dataset (same seed/split as training)
    print("\n[1/4] Loading dataset...")
    _, test_products = load_products(seed=42)
    test_products = test_products[:N_TEST]
    mi_count = sum(1 for p in test_products if p["mi"])
    ci_count = len(test_products) - mi_count
    print(f"  Using {len(test_products)} test products (MI={mi_count}, CI={ci_count})")
    
    # Load models
    print(f"\n[2/4] Loading buyer model: {BUYER_PATH}" + (f" @ {BUYER_REV}" if BUYER_REV else "") + "...")
    buyer_stack = load_eval_stack(BUYER_PATH, BUYER_REV)
    print(f"  [OK] image_text={buyer_stack['is_image_text']} VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Seller model (frozen baseline)
    same_model = (SELLER_PATH == BUYER_PATH and not BUYER_REV)
    if same_model:
        print(f"\n[3/4] Seller = same as buyer (self-play eval)")
        seller_stack = buyer_stack
    else:
        print(f"\n[3/4] Loading seller model: {SELLER_PATH}...")
        seller_stack = load_eval_stack(SELLER_PATH)
    print(f"  [OK] image_text={seller_stack['is_image_text']} VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Run evaluation
    print(f"\n[4/4] Running {N_TEST * N_ROLL} evaluation episodes...")
    print(f"{'─' * 70}")
    
    all_episodes = []
    t0 = time.time()
    
    for prod_idx, product in enumerate(test_products):
        for roll in range(N_ROLL):
            ep = run_eval_episode(buyer_stack, seller_stack, product)
            all_episodes.append(ep)
        
        # Progress logging every 16 products
        if (prod_idx + 1) % 16 == 0 or prod_idx == len(test_products) - 1:
            done = (prod_idx + 1) * N_ROLL
            elapsed = time.time() - t0
            rate = elapsed / done
            eta = rate * (N_TEST * N_ROLL - done)
            partial = compute_metrics(all_episodes, "partial")
            print(f"  [{done}/{N_TEST*N_ROLL}] R={partial['reward']:+.4f} "
                  f"Deal={partial['deal_rate']:.1%} BR={partial['bargained_ratio']:.4f} "
                  f"BV={partial['price_overshoot_rate']:.1%} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
    
    total_time = time.time() - t0
    print(f"{'─' * 70}")
    print(f"Completed {len(all_episodes)} episodes in {total_time:.1f}s ({total_time/60:.1f}min)\n")
    
    # ─── Compute all metrics ──────────────────────────────────────────────────
    
    # Overall
    overall = compute_metrics(all_episodes, "overall")
    
    # MI/CI split (paper Appendix D.4)
    mi_episodes = [ep for ep in all_episodes if ep.product["mi"]]
    ci_episodes = [ep for ep in all_episodes if not ep.product["mi"]]
    mi_metrics = compute_metrics(mi_episodes, "MI")
    ci_metrics = compute_metrics(ci_episodes, "CI")
    
    # ─── Print results ────────────────────────────────────────────────────────
    print(f"{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    
    def print_metrics(m):
        print(f"  [{m['label']}] n={m['n']}")
        print(f"    Reward:             {m['reward']:+.4f} ± {m['reward_se']:.4f}")
        print(f"    Deal Rate:          {m['deal_rate']:.1%}")
        print(f"    Bargained Ratio:    {m['bargained_ratio']:.4f}")
        if m.get('first_offer_ratio') is not None:
            print(f"    First Offer Ratio:  {m['first_offer_ratio']:.4f}")
        print(f"    Price Overshoot:    {m['price_overshoot_rate']:.1%}")
        print(f"    Mean Turns:         {m['mean_turns']:.2f}")
        print(f"    Outcomes:           {m['outcomes']}")
    
    print_metrics(overall)
    print()
    print_metrics(mi_metrics)
    print()
    print_metrics(ci_metrics)
    
    # Paper comparison table
    print(f"\n{'─' * 70}")
    print(f"Paper Comparison (Table 1 — held-out test, neutral seller)")
    print(f"{'─' * 70}")
    print(f"  {'Model':<40} {'Reward':>8} {'Deal%':>7} {'BR':>7} {'BV%':>6}")
    print(f"  {'─'*40} {'─'*8} {'─'*7} {'─'*7} {'─'*6}")
    print(f"  {'Paper 30B-A3B (trained, 60 iters)':<40} {'+0.7664':>8} {'92.0%':>7} {'0.8385':>7} {'0.1%':>6}")
    print(f"  {'GPT-5.4-high-reasoning':<40} {'+0.4021':>8} {'91.8%':>7} {'0.4380':>7} {'0.0%':>6}")
    print(f"  {'Qwen3-4B-Instruct-2507 (untrained)':<40} {'-0.1130':>8} {'51.6%':>7} {'0.0901':>7} {'6.5%':>6}")
    ours_label = f"Ours ({BUYER_REV or 'main'})"
    print(f"  {ours_label:<40} {overall['reward']:>+8.4f} {overall['deal_rate']:>6.1%} {overall['bargained_ratio']:>7.4f} {overall['price_overshoot_rate']:>5.1%}")
    print(f"{'─' * 70}\n")
    
    # ─── Save results ─────────────────────────────────────────────────────────
    results = {
        "config": {
            "buyer_path": BUYER_PATH,
            "buyer_rev": BUYER_REV,
            "seller_path": SELLER_PATH,
            "n_test": N_TEST,
            "n_roll": N_ROLL,
            "max_turns": MAX_TURNS,
            "buyer_temp": BUYER_TEMP,
            "seller_temp": SELLER_TEMP,
            "max_new": MAX_NEW,
            "seed": SEED,
        },
        "overall": overall,
        "mi": mi_metrics,
        "ci": ci_metrics,
        "total_time_s": round(total_time, 1),
        # Per-episode details for deeper analysis
        "episodes": [
            {
                "product": ep.product["codename"],
                "mi": ep.product["mi"],
                "reward": round(ep.reward, 4),
                "final_price": ep.final_price,
                "budget": ep.product["budget"],
                "cost": ep.product["cost"],
                "outcome": ep.outcome,
                "num_turns": ep.num_turns,
                "first_offer": ep.first_offer_price,
                "budget_violations": ep.budget_violations,
            }
            for ep in all_episodes
        ],
    }
    
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUT}")
    
    # Push to hub if configured
    if HUB_MODEL_ID:
        try:
            from huggingface_hub import HfApi
            token = os.environ.get("HF_TOKEN")
            api = HfApi(token=token)
            rev = BUYER_REV or "main"
            remote_path = f"eval_results_{rev}.json"
            api.upload_file(
                path_or_fileobj=OUT, path_in_repo=remote_path,
                repo_id=HUB_MODEL_ID, repo_type="model",
                commit_message=f"Eval {rev}: R={overall['reward']:+.4f} Deal={overall['deal_rate']:.1%}",
            )
            print(f"Pushed to {HUB_MODEL_ID}/{remote_path}")
        except Exception as e:
            print(f"[WARN] Hub push failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
