"""
GRPO Training for Bilateral Negotiation (Paper 2604.09855 replication).
H200 GPU. Full fine-tuning. Same model for buyer (trainable) and seller (frozen).
No LoRA. No TRL dependency. Custom multi-turn GRPO loop.

Env vars:
    MODEL_NAME      Qwen model ID (default: Qwen/Qwen3-4B)
    NUM_ITERS       Training iterations (default: 5 for toy, 40 for real)
    BATCH_SIZE      Products per iteration (default: 16)
    GROUP_SIZE      Rollouts per product (default: 8)
    MAX_TURNS       Max negotiation turns (default: 6)
    LR              Learning rate (default: 3e-5)
    OUTPUT_DIR      Where to save model (default: /tmp/model)
    HUB_MODEL_ID    HF Hub repo to push to (optional)
"""
import os
import sys
import re
import json
import random
import time
import math
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")
NUM_ITERS = int(os.environ.get("NUM_ITERS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))
LR = float(os.environ.get("LR", "3e-5"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/model")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "150"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
SELLER_TEMP = float(os.environ.get("SELLER_TEMP", "0.7"))
KL_COEF = float(os.environ.get("KL_COEF", "0.0"))
EPSILON = float(os.environ.get("EPSILON", "0.2"))
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1"


# ─── Diagnostics ──────────────────────────────────────────────────────────────
def log_cuda():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FATAL: No CUDA available")
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    x = torch.randn(2, 2).cuda() @ torch.randn(2, 2).cuda()
    print(f"CUDA tensor test: {x.device} OK")
    print("=" * 60)


# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_BASE = (
    "https://raw.githubusercontent.com/TianXiaSJTU/AmazonPriceHistory"
    "/main/data/AmazonHistoryPrice/"
)
CATEGORIES = [
    "automotive", "baby-products", "beauty", "books", "electronics",
    "health-personal-care", "home-kitchen", "industrial-scientific",
    "movies-tv", "music", "other", "patio-lawn-garden", "pet-supplies",
    "software", "sports-outdoors", "tools-home-improvement",
]


def parse_price(s: str) -> float:
    return float(s.replace("$", "").replace(",", "").strip())


def load_products(seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Download and parse AmazonHistoryPrice. Return (train, test)."""
    import urllib.request

    all_items = []
    for cat in CATEGORIES:
        url = DATASET_BASE + f"{cat}.json"
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
    print(f"[DATA] Products: {len(all_items)}  train={len(train)}  test={len(test)}  MI={mi}  CI={len(all_items)-mi}")
    return train, test


# ─── Prompts (from paper / original repo) ───────────────────────────────────
BUYER_SYSTEM_PROMPT = """You are a buyer looking forward to buying things on your Shopping List from me, the seller.
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


SELLER_SYSTEM_PROMPT = """You are a seller looking forward to selling things on your Inventory List to me, the buyer.
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


def build_buyer_prompt(product: dict) -> List[dict]:
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
        {"role": "system", "content": BUYER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_seller_prompt(product: dict, buyer_history: List[str]) -> List[dict]:
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
    messages = [
        {"role": "system", "content": SELLER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    for i, msg in enumerate(buyer_history):
        role = "assistant" if i % 2 == 0 else "user"
        messages.append({"role": role, "content": msg})
    return messages


# ─── Action Extraction ───────────────────────────────────────────────────────
ACTION_RE = re.compile(
    r'\[(BUY|SELL|DEAL|REJECT|QUIT)\]'
    r'(?:\s*\$([\d,\.]+))?'
    r'(?:\s*\(([^)]*)\))?',
    re.IGNORECASE,
)


def extract_action(text: str) -> dict:
    m = ACTION_RE.search(text)
    if m:
        price_str = m.group(2)
        price = float(price_str.replace(",", "")) if price_str else None
        return {"type": m.group(1).upper(), "price": price, "objects": m.group(3), "raw": m.group(0)}
    return {"type": "UNKNOWN", "price": None, "objects": None, "raw": text[-100:]}


# ─── Seller Regulation ────────────────────────────────────────────────────────
def regulated_seller_responds(seller_action: dict, buyer_price: Optional[float], product: dict) -> Tuple[Optional[float], bool, str]:
    """Regulate seller: cannot accept below cost, cannot propose below cost."""
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
            return None, True, f"SELLER_CANNOT_ACCEPT_BELOW_COST ({buyer_price} < {cost})"
        return buyer_price, True, "DEAL"

    if at == "SELL":
        if price is None:
            return None, True, "NO_PRICE_IN_SELL"
        if price < cost:
            # Regulate: seller cannot offer below cost
            price = cost * 1.05  # Minimum markup
        return price, False, "SELL"

    if at == "REJECT":
        return None, False, "REJECT"

    return None, True, f"UNEXPECTED_ACTION_{at}"


# ─── Reward ───────────────────────────────────────────────────────────────────
def compute_reward(final_price: Optional[float], budget: float, cost: float) -> float:
    """Paper's reward formula."""
    if final_price is None:
        return 0.0  # No deal / QUIT

    denom = abs(budget - cost)
    if denom < 1e-6:
        return 0.0

    # Budget violation check
    if final_price > budget:
        return -1.0

    reward = (budget - final_price) / denom
    return max(-1.0, min(1.0, reward))


# ─── Generation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_response(model, tokenizer, messages: List[dict], max_new: int, temp: float, device: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Generate one response. Returns (text, token_ids, token_logprobs)."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temp,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    gen_tokens = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # Compute token logprobs
    scores = torch.stack(outputs.scores, dim=1)  # [1, gen_len, vocab]
    log_probs = F.log_softmax(scores, dim=-1)
    token_logprobs = torch.gather(log_probs[0], 1, gen_tokens.unsqueeze(-1)).squeeze(-1)

    return gen_text, gen_tokens, token_logprobs


# ─── Episode Runner ───────────────────────────────────────────────────────────
@dataclass
class Episode:
    product: dict
    buyer_messages: List[Tuple[str, torch.Tensor, torch.Tensor]]  # (text, tokens, logprobs)
    seller_messages: List[str]
    final_price: Optional[float]
    reward: float
    turns: int
    outcome: str


def run_episode(buyer_model, seller_model, tokenizer, product: dict, device: str) -> Episode:
    """Run one complete negotiation episode."""
    buyer_prompt = build_buyer_prompt(product)
    buyer_history = []
    seller_history = []
    final_price = None
    outcome = "TIMEOUT"

    for turn in range(MAX_TURNS):
        # Buyer turn
        buyer_text, buyer_tokens, buyer_logprobs = generate_response(
            buyer_model, tokenizer, buyer_prompt + buyer_history,
            max_new=MAX_NEW_TOKENS, temp=TEMPERATURE, device=device,
        )
        buyer_action = extract_action(buyer_text)
        buyer_history.append({"role": "assistant", "content": buyer_text})

        # Check buyer terminal actions
        if buyer_action["type"] == "QUIT":
            outcome = "BUYER_QUIT"
            break
        if buyer_action["type"] == "UNKNOWN":
            outcome = "BUYER_FORMAT_ERROR"
            final_price = None
            break
        if buyer_action["type"] == "DEAL":
            # Buyer accepting seller's previous offer
            if turn == 0 or not seller_history:
                outcome = "BUYER_DEAL_WITHOUT_SELLER_OFFER"
                final_price = None
                break
            # Use the last seller's offered price
            last_seller = extract_action(seller_history[-1])
            final_price = last_seller.get("price")
            outcome = "DEAL_BUYER_ACCEPTS"
            break
        if buyer_action["type"] == "BUY":
            buyer_price = buyer_action["price"]
            if buyer_price is not None and buyer_price > product["budget"]:
                outcome = "BUYER_BUDGET_VIOLATION"
                final_price = None
                break

            # Seller turn
            seller_prompt = build_seller_prompt(product, [buyer_text])
            seller_text, _, _ = generate_response(
                seller_model, tokenizer, seller_prompt,
                max_new=MAX_NEW_TOKENS, temp=SELLER_TEMP, device=device,
            )
            seller_action = extract_action(seller_text)
            seller_history.append(seller_text)
            buyer_history.append({"role": "user", "content": seller_text})

            # Regulate seller
            regulated_price, done, reason = regulated_seller_responds(seller_action, buyer_price, product)

            if done:
                if reason.startswith("DEAL"):
                    final_price = regulated_price
                    outcome = "DEAL_SELLER_ACCEPTS"
                elif reason == "QUIT":
                    outcome = "SELLER_QUIT"
                elif reason.startswith("SELLER_CANNOT_ACCEPT"):
                    outcome = reason
                    final_price = None
                elif reason == "FORMAT_ERROR":
                    outcome = "SELLER_FORMAT_ERROR"
                    final_price = None
                else:
                    outcome = reason
                    final_price = None
                break

            # Continue negotiation
            if seller_action["type"] == "SELL" and regulated_price is not None:
                # Update prompt with seller's counter
                pass  # Already appended above
        else:
            # REJECT or other non-terminal
            # Seller gets to respond
            seller_prompt = build_seller_prompt(product, [buyer_text])
            seller_text, _, _ = generate_response(
                seller_model, tokenizer, seller_prompt,
                max_new=MAX_NEW_TOKENS, temp=SELLER_TEMP, device=device,
            )
            seller_history.append(seller_text)
            buyer_history.append({"role": "user", "content": seller_text})

    reward = compute_reward(final_price, product["budget"], product["cost"])

    # Build episode with only buyer messages (these are what we train on)
    buyer_msg_list = []
    for msg in buyer_history:
        if msg.get("role") == "assistant":
            # We need to regenerate to get tokens/logprobs for storage
            # But we already have them from the generate step above
            # For simplicity, we'll store the text and re-tokenize later for loss
            buyer_msg_list.append((msg["content"], None, None))

    return Episode(
        product=product,
        buyer_messages=buyer_msg_list,
        seller_messages=seller_history,
        final_price=final_price,
        reward=reward,
        turns=turn,
        outcome=outcome,
    )


# ─── GRPO Update ──────────────────────────────────────────────────────────────
def grpo_update(buyer_model, ref_model, tokenizer, episodes: List[Episode], optimizer, device: str) -> float:
    """GRPO loss: per-group mean-baseline advantage, clipped ratio."""
    buyer_model.train()

    # Group episodes by product
    # episodes are ordered: [p0_g0, p0_g1, ..., p0_gG-1, p1_g0, ...]
    G = GROUP_SIZE
    num_groups = len(episodes) // G

    total_loss = 0.0
    num_tokens = 0

    for g in range(num_groups):
        group_eps = episodes[g * G : (g + 1) * G]
        rewards = torch.tensor([ep.reward for ep in group_eps], dtype=torch.float32, device=device)
        mean_reward = rewards.mean()
        advantages = rewards - mean_reward

        for i, ep in enumerate(group_eps):
            # Reconstruct full prompt + completion
            prompt_messages = build_buyer_prompt(ep.product)
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )

            # Get buyer's actual response text
            # We need to regenerate or store the tokens. For simplicity, we'll tokenize the text.
            # But we need per-token logprobs for the GRPO loss.
            # The proper way: during episode generation, store the generated tokens and their logprobs.
            # Then reconstruct the sequence for the forward pass.

            # For now, we'll tokenize the buyer's first message and compute loss on it
            if not ep.buyer_messages:
                continue
            buyer_text = ep.buyer_messages[0][0]
            full_text = prompt_text + buyer_text

            full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(device)["input_ids"]
            prompt_len = prompt_ids.shape[1]

            # Policy forward
            outputs = buyer_model(**full_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            token_log_probs = torch.gather(log_probs, 2, full_ids["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)

            # Reference forward
            with torch.no_grad():
                ref_outputs = ref_model(**full_ids)
                ref_logits = ref_outputs.logits
                ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                ref_token_log_probs = torch.gather(ref_log_probs, 2, full_ids["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)

            # Completion mask
            mask = full_ids["attention_mask"][:, 1:].clone()
            mask[:, :prompt_len - 1] = 0  # Only completion tokens

            # Ratio
            ratio = torch.exp(token_log_probs - ref_token_log_probs)
            clipped = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)

            # GRPO surrogate
            surr1 = ratio * advantages[i]
            surr2 = clipped * advantages[i]
            policy_loss = -torch.min(surr1, surr2)

            # KL penalty (paper uses 0, but keep the code)
            if KL_COEF > 0:
                kl = token_log_probs - ref_token_log_probs
                policy_loss = policy_loss + KL_COEF * kl

            # Masked mean
            loss = (policy_loss * mask).sum() / (mask.sum() + 1e-8)
            total_loss += loss
            num_tokens += mask.sum().item()

    if num_groups == 0:
        return 0.0

    avg_loss = total_loss / num_groups
    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(buyer_model.parameters(), 1.0)
    optimizer.step()

    return avg_loss.item()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    log_cuda()

    print(f"[CONFIG] Model: {MODEL_NAME}")
    print(f"[CONFIG] Iters: {NUM_ITERS}  Batch: {BATCH_SIZE}  Group: {GROUP_SIZE}  MaxTurns: {MAX_TURNS}")
    print(f"[CONFIG] LR: {LR}  KL: {KL_COEF}  Eps: {EPSILON}  Temp: {TEMPERATURE}/{SELLER_TEMP}")
    print(f"[CONFIG] Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
    print("=" * 60)

    # Dataset
    print("\n[1/5] Loading dataset...")
    train_products, test_products = load_products()

    # Tokenizer
    print(f"\n[2/5] Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  [OK]")

    # Buyer model (trainable)
    print(f"\n[3/5] Loading buyer model...")
    buyer_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if GRADIENT_CHECKPOINTING:
        buyer_model.gradient_checkpointing_enable()
    print(f"  [OK] Device: {next(buyer_model.parameters()).device}")

    # Reference / Seller model (frozen copy)
    print(f"\n[4/5] Loading reference/seller model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  [OK] Device: {next(ref_model.parameters()).device}")

    # Seller model = same frozen copy
    seller_model = ref_model

    # Optimizer (full fine-tuning)
    print(f"\n[5/5] Setting up optimizer (AdamW, lr={LR})...")
    optimizer = torch.optim.AdamW(buyer_model.parameters(), lr=LR)
    print(f"  Trainable params: {sum(p.numel() for p in buyer_model.parameters() if p.requires_grad):,}")

    # Training loop
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print(f"{'=' * 60}")

    metrics_log = []
    t0 = time.time()

    for iteration in range(NUM_ITERS):
        t1 = time.time()
        print(f"\n--- Iteration {iteration} ---")

        # Sample products
        products = random.sample(train_products, min(BATCH_SIZE, len(train_products)))

        # Generate episodes (BATCH_SIZE × GROUP_SIZE total)
        print(f"  Generating {len(products)} × {GROUP_SIZE} = {len(products)*GROUP_SIZE} episodes...")
        episodes = []
        for p in products:
            for g in range(GROUP_SIZE):
                ep = run_episode(buyer_model, seller_model, tokenizer, p, device=buyer_model.device)
                episodes.append(ep)

        # GRPO update
        print("  Computing GRPO loss and updating...")
        loss = grpo_update(buyer_model, ref_model, tokenizer, episodes, optimizer, device=buyer_model.device)

        # Metrics
        rewards = [ep.reward for ep in episodes]
        mean_reward = sum(rewards) / len(rewards)
        deal_count = sum(1 for ep in episodes if ep.final_price is not None)
        deal_rate = deal_count / len(episodes)
        fp_vals = [ep.final_price for ep in episodes if ep.final_price is not None]
        mean_price = sum(fp_vals) / max(1, len(fp_vals))
        mean_turns = sum(ep.turns for ep in episodes) / len(episodes)

        # Outcome distribution
        outcomes = {}
        for ep in episodes:
            outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1

        elapsed = time.time() - t1
        print(f"  Loss: {loss:.4f}  Reward: {mean_reward:.4f}  Deal: {deal_rate:.1%}  "
              f"Price: ${mean_price:.2f}  Turns: {mean_turns:.1f}  Time: {elapsed:.1f}s")
        print(f"  Outcomes: {dict(sorted(outcomes.items(), key=lambda x: -x[1])[:5])}")

        # Sample episode
        sample = episodes[0]
        print(f"  Sample: {sample.outcome}  Price: ${sample.final_price}  Reward: {sample.reward:.4f}")

        metrics_log.append({
            "iteration": iteration,
            "loss": loss,
            "mean_reward": mean_reward,
            "deal_rate": deal_rate,
            "mean_price": mean_price,
            "mean_turns": mean_turns,
            "time": elapsed,
            "outcomes": outcomes,
        })

    # Save
    print(f"\n{'=' * 60}")
    print("SAVING")
    print(f"{'=' * 60}")
    save_path = Path(OUTPUT_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    buyer_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"  Model saved to {save_path}")

    # Push to hub
    if HUB_MODEL_ID:
        try:
            from huggingface_hub import HfApi, create_repo
            print(f"\nPushing to {HUB_MODEL_ID}...")
            create_repo(HUB_MODEL_ID, exist_ok=True)
            HfApi().upload_folder(folder_path=save_path, repo_id=HUB_MODEL_ID, repo_type="model")
            print("  [OK] Push complete")
        except Exception as e:
            print(f"  [WARN] Push failed: {e}")

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"COMPLETE  Total time: {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
