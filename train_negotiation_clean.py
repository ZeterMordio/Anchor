"""
GRPO Training for Bilateral Negotiation
Replication of https://huggingface.co/papers/2604.09855

Architecture:
- Qwen3-4B/8B as buyer (trainable) + Qwen3-4B/8B as seller (frozen, same model)
- Full fine-tuning (no LoRA — 143GB H200 has plenty VRAM)
- Custom multi-turn GRPO loop (not TRL — their GRPO doesn't support multi-turn environments)
- Paper-prompts, paper-reward formula, paper-seller regulation

Toy Run 1: 5 iters, Qwen3-4B, buyer-only training, verify pipeline
Toy Run 2: 15 iters, Qwen3-4B, buyer-only, check for 4-phase convergence
Toy Run 3: 15 iters, Qwen3-4B, dual-role + RAE
Real Run: 40-60 iters, Qwen3-8B, buyer-only or dual-role per toy results
"""

import os
import sys
import re
import json
import random
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Config from env ─────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")
NUM_ITERS = int(os.environ.get("NUM_ITERS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))   # products per iter
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))     # rollouts per product
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))
LR = float(os.environ.get("LR", "3e-5"))
EPSILON = float(os.environ.get("EPSILON", "0.2"))
KL_COEF = float(os.environ.get("KL_COEF", "0.0"))       # paper uses 0
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "150"))
BUYER_TEMP = float(os.environ.get("BUYER_TEMP", "1.0"))
SELLER_TEMP = float(os.environ.get("SELLER_TEMP", "0.7"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/model")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1"

# ─── CUDA check ────────────────────────────────────────────────────────────────
def check_cuda():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FATAL: No CUDA")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Quick compute test
    x = torch.randn(2, 2).cuda() @ torch.randn(2, 2).cuda()
    print(f"Compute test: {x.device} OK")
    print("=" * 60)

# ─── Dataset: AmazonHistoryPrice ──────────────────────────────────────────────
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

# ─── Prompts (exactly from paper / original repo) ─────────────────────────────
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

def build_seller_prompt(product, buyer_history_texts):
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
        {"role": "system", "content": SELLER_SYSTEM},
        {"role": "user", "content": user},
    ]
    for i, msg in enumerate(buyer_history_texts):
        role = "assistant" if i % 2 == 0 else "user"
        messages.append({"role": role, "content": msg})
    return messages

# ─── Action extraction ─────────────────────────────────────────────────────────
ACTION_RE = re.compile(
    r'\[(BUY|SELL|DEAL|REJECT|QUIT)\]'
    r'(?:\s*\$([\d,\.]+))?'
    r'(?:\s*\(([^)]*)\))?',
    re.IGNORECASE,
)

def extract_action(text):
    m = ACTION_RE.search(text)
    if m:
        ps = m.group(2)
        price = float(ps.replace(",", "")) if ps else None
        return {"type": m.group(1).upper(), "price": price, "objects": m.group(3)}
    return {"type": "UNKNOWN", "price": None, "objects": None}

# ─── Seller regulation (paper §3.2) ────────────────────────────────────────────
def regulate_seller(seller_action, buyer_price, product):
    """Regulate seller per paper: cannot accept below cost, cannot propose below cost."""
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
            return None, True, f"SELLER_CANNOT_ACCEPT_BELOW_COST"
        return buyer_price, True, "DEAL"
    if at == "SELL":
        if price is None:
            return None, True, "NO_PRICE_IN_SELL"
        if price < cost:
            price = cost * 1.05  # Regulate: minimum markup
        return price, False, "SELL"
    if at == "REJECT":
        return None, False, "REJECT"
    return None, True, f"UNEXPECTED_{at}"

# ─── Reward formula (paper §3.2, Eq 1) ─────────────────────────────────────────
def compute_reward(final_price, budget, cost):
    if final_price is None:
        return 0.0  # No deal / QUIT
    if final_price > budget:
        return -1.0  # Budget violation
    denom = abs(budget - cost)
    if denom < 1e-6:
        return 0.0
    r = (budget - final_price) / denom
    return max(-1.0, min(1.0, r))

# ─── Generation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, messages, max_new, temp, device):
    """Generate response. Returns (text, token_ids, token_logprobs)."""
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

    # Token logprobs
    scores = torch.stack(outputs.scores, dim=1)  # [1, gen_len, vocab]
    log_probs = F.log_softmax(scores, dim=-1)
    token_logprobs = torch.gather(
        log_probs[0], 1, gen_tokens.unsqueeze(-1)
    ).squeeze(-1)

    return gen_text, gen_tokens, token_logprobs

# ─── Episode runner ────────────────────────────────────────────────────────────
@dataclass
class Episode:
    product: dict
    buyer_turns: List[Tuple[str, torch.Tensor, torch.Tensor]]  # (text, tokens, logprobs)
    seller_turns: List[str]
    final_price: Optional[float]
    reward: float
    turns: int
    outcome: str

def run_episode(buyer_model, seller_model, tokenizer, product, device):
    """Run one complete negotiation. Returns Episode."""
    buyer_prompt = build_buyer_prompt(product)
    buyer_history = []      # list of {"role": "assistant"/"user", "content": str}
    buyer_texts = []        # only buyer texts for seller prompt
    seller_history = []
    final_price = None
    outcome = "TIMEOUT"

    for turn in range(MAX_TURNS):
        # Buyer generates
        msgs = buyer_prompt + buyer_history
        b_text, b_tokens, b_logprobs = generate(
            buyer_model, tokenizer, msgs, MAX_NEW_TOKENS, BUYER_TEMP, device
        )
        b_act = extract_action(b_text)
        buyer_history.append({"role": "assistant", "content": b_text})
        buyer_texts.append(b_text)

        # Terminal buyer actions
        if b_act["type"] == "QUIT":
            outcome = "BUYER_QUIT"
            break
        if b_act["type"] == "UNKNOWN":
            outcome = "BUYER_FORMAT_ERROR"
            final_price = None
            break
        if b_act["type"] == "DEAL":
            if turn == 0 or not seller_history:
                outcome = "BUYER_DEAL_NO_SELLER_OFFER"
                final_price = None
                break
            last_seller = extract_action(seller_history[-1])
            final_price = last_seller.get("price")
            outcome = "DEAL_BUYER_ACCEPTS"
            break
        if b_act["type"] == "BUY":
            b_price = b_act["price"]
            if b_price is not None and b_price > product["budget"]:
                outcome = "BUYER_BUDGET_VIOLATION"
                final_price = None
                break

            # Seller turn
            s_msgs = build_seller_prompt(product, buyer_texts)
            s_text, _, _ = generate(
                seller_model, tokenizer, s_msgs, MAX_NEW_TOKENS, SELLER_TEMP, device
            )
            s_act = extract_action(s_text)
            seller_history.append(s_text)
            buyer_history.append({"role": "user", "content": s_text})
            buyer_texts.append(s_text)

            # Regulate seller
            r_price, done, reason = regulate_seller(s_act, b_price, product)

            if done:
                if reason == "DEAL":
                    final_price = r_price
                    outcome = "DEAL_SELLER_ACCEPTS"
                elif reason == "QUIT":
                    outcome = "SELLER_QUIT"
                elif "BELOW_COST" in reason:
                    outcome = reason
                    final_price = None
                elif reason == "FORMAT_ERROR":
                    outcome = "SELLER_FORMAT_ERROR"
                    final_price = None
                else:
                    outcome = reason
                    final_price = None
                break
            # Continue — seller's counter is now in buyer_history for next turn
        else:
            # REJECT — seller gets to respond
            s_msgs = build_seller_prompt(product, buyer_texts)
            s_text, _, _ = generate(
                seller_model, tokenizer, s_msgs, MAX_NEW_TOKENS, SELLER_TEMP, device
            )
            s_act = extract_action(s_text)
            seller_history.append(s_text)
            buyer_history.append({"role": "user", "content": s_text})
            buyer_texts.append(s_text)
            # Continue to next turn

    reward = compute_reward(final_price, product["budget"], product["cost"])

    # Build episode: store buyer turn data
    buyer_turn_data = []
    for msg in buyer_history:
        if msg.get("role") == "assistant":
            # We have tokens/logprobs from generation — but only stored for last turn
            # For simplicity, we'll re-tokenize all buyer turns during GRPO update
            buyer_turn_data.append((msg["content"], None, None))

    return Episode(
        product=product,
        buyer_turns=buyer_turn_data,
        seller_turns=seller_history,
        final_price=final_price,
        reward=reward,
        turns=turn,
        outcome=outcome,
    )

# ─── GRPO Update ───────────────────────────────────────────────────────────────
def grpo_update(buyer_model, ref_model, tokenizer, episodes, optimizer, device):
    """
    GRPO per paper §3.2:
    - Group episodes by product (G rollouts per product)
    - Advantage = reward - group_mean_reward
    - Clipped PPO-style ratio with per-token logprobs
    """
    buyer_model.train()
    G = GROUP_SIZE
    num_groups = len(episodes) // G

    for g in range(num_groups):
        group_eps = episodes[g * G : (g + 1) * G]
        rewards = torch.tensor(
            [ep.reward for ep in group_eps], dtype=torch.float32, device=device
        )
        mean_r = rewards.mean()
        advantages = rewards - mean_r

        group_loss = 0.0

        for i, ep in enumerate(group_eps):
            # For each buyer turn in the episode, compute loss
            # We train on ALL buyer turns (multi-turn GRPO)
            for turn_idx, (b_text, _, _) in enumerate(ep.buyer_turns):
                # Build prompt up to this turn
                prompt_msgs = build_buyer_prompt(ep.product)
                # Add conversation history before this turn
                for j, msg in enumerate(ep.buyer_turns[:turn_idx]):
                    # Add seller response between turns (from seller_turns)
                    if j < len(ep.seller_turns):
                        prompt_msgs.append({"role": "user", "content": ep.seller_turns[j]})
                    prompt_msgs.append({"role": "assistant", "content": msg[0]})
                # Current turn's prompt ends before buyer response
                if turn_idx < len(ep.seller_turns) and turn_idx > 0:
                    prompt_msgs.append({"role": "user", "content": ep.seller_turns[turn_idx - 1]})

                prompt_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                full_text = prompt_text + b_text

                full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
                prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(device)["input_ids"]
                prompt_len = prompt_ids.shape[1]

                # Policy forward
                out = buyer_model(**full_ids)
                logits = out.logits
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                tok_log_probs = torch.gather(
                    log_probs, 2, full_ids["input_ids"][:, 1:].unsqueeze(-1)
                ).squeeze(-1)

                # Reference forward
                with torch.no_grad():
                    ref_out = ref_model(**full_ids)
                    ref_logits = ref_out.logits
                    ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                    ref_tok_log_probs = torch.gather(
                        ref_log_probs, 2, full_ids["input_ids"][:, 1:].unsqueeze(-1)
                    ).squeeze(-1)

                # Completion mask
                mask = full_ids["attention_mask"][:, 1:].clone()
                mask[:, :prompt_len - 1] = 0

                # Ratio & clip
                ratio = torch.exp(tok_log_probs - ref_tok_log_probs)
                clipped = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)

                surr1 = ratio * advantages[i]
                surr2 = clipped * advantages[i]
                policy_loss = -torch.min(surr1, surr2)

                if KL_COEF > 0:
                    kl = tok_log_probs - ref_tok_log_probs
                    policy_loss = policy_loss + KL_COEF * kl

                loss = (policy_loss * mask).sum() / (mask.sum() + 1e-8)
                loss.backward()
                group_loss += loss.item()

        # After processing all episodes in the group, step optimizer
        torch.nn.utils.clip_grad_norm_(buyer_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if num_groups == 0:
        return 0.0

    return group_loss / (num_groups * G)  # average per episode

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    check_cuda()

    print(f"[CONFIG] Model={MODEL_NAME} Iters={NUM_ITERS} Batch={BATCH_SIZE} Group={GROUP_SIZE}")
    print(f"[CONFIG] Turns={MAX_TURNS} LR={LR} Eps={EPSILON} KL={KL_COEF}")
    print(f"[CONFIG] BuyerTemp={BUYER_TEMP} SellerTemp={SELLER_TEMP} MaxNew={MAX_NEW_TOKENS}")
    print(f"[CONFIG] GradCheckpoint={GRADIENT_CHECKPOINTING}")
    print("=" * 60)

    # 1. Dataset
    print("\n[1/5] Loading dataset...")
    train_products, test_products = load_products()

    # 2. Tokenizer
    print(f"\n[2/5] Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  [OK]")

    # 3. Buyer model (trainable)
    print(f"\n[3/5] Loading buyer model...")
    buyer_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if GRADIENT_CHECKPOINTING:
        buyer_model.gradient_checkpointing_enable()
    dev = next(buyer_model.parameters()).device
    print(f"  [OK] Device={dev}")

    # 4. Reference / Seller model (frozen, same model)
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
    print(f"  [OK] Device={next(ref_model.parameters()).device}")
    seller_model = ref_model

    # 5. Optimizer
    print(f"\n[5/5] Optimizer (AdamW, lr={LR})...")
    optimizer = torch.optim.AdamW(buyer_model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in buyer_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # Training
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print(f"{'=' * 60}")

    metrics = []
    t0 = time.time()

    for iteration in range(NUM_ITERS):
        t1 = time.time()
        print(f"\n--- Iteration {iteration} ---")

        products = random.sample(train_products, min(BATCH_SIZE, len(train_products)))
        print(f"  Sampling {len(products)} products, {GROUP_SIZE} rollouts each...")

        episodes = []
        for p in products:
            for _ in range(GROUP_SIZE):
                ep = run_episode(buyer_model, seller_model, tokenizer, p, dev)
                episodes.append(ep)

        print(f"  GRPO update on {len(episodes)} episodes...")
        loss = grpo_update(buyer_model, ref_model, tokenizer, episodes, optimizer, dev)

        # Metrics
        rewards = [ep.reward for ep in episodes]
        mean_r = sum(rewards) / len(rewards)
        deals = sum(1 for ep in episodes if ep.final_price is not None)
        deal_rate = deals / len(episodes)
        fp = [ep.final_price for ep in episodes if ep.final_price is not None]
        mean_price = sum(fp) / max(1, len(fp))
        mean_turns = sum(ep.turns for ep in episodes) / len(episodes)

        outcomes = {}
        for ep in episodes:
            outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1

        elapsed = time.time() - t1
        print(f"  Loss={loss:.4f} Reward={mean_r:.4f} Deal={deal_rate:.1%} "
              f"Price=${mean_price:.2f} Turns={mean_turns:.1f} Time={elapsed:.1f}s")
        print(f"  Outcomes: {dict(sorted(outcomes.items(), key=lambda x: -x[1])[:5])}")

        # Sample
        sample = episodes[0]
        print(f"  Sample: {sample.outcome} Price={sample.final_price} Reward={sample.reward:.4f}")

        metrics.append({
            "iteration": iteration,
            "loss": loss,
            "mean_reward": mean_r,
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
        json.dump(metrics, f, indent=2)
    print(f"  Saved to {save_path}")

    # Push to hub
    if HUB_MODEL_ID:
        try:
            from huggingface_hub import HfApi, create_repo
            token = os.environ.get("HF_TOKEN")
            print(f"\nPushing to {HUB_MODEL_ID}...")
            create_repo(HUB_MODEL_ID, exist_ok=True, token=token)
            HfApi(token=token).upload_folder(folder_path=save_path, repo_id=HUB_MODEL_ID, repo_type="model")
            print("  [OK] Push complete")
        except Exception as e:
            print(f"  [WARN] Push failed: {e}")

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"COMPLETE  Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
