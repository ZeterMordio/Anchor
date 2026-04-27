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

Toy Run 3: 15 iters, Qwen3-4B, dual-role + RAE
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

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")
NUM_ITERS = int(os.environ.get("NUM_ITERS", "15"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))
LR = float(os.environ.get("LR", "3e-5"))
EPSILON = float(os.environ.get("EPSILON", "0.2"))
KL_COEF = float(os.environ.get("KL_COEF", "0.0"))
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


# ─── Generation (FIXED: no output_scores leak) ────────────────────────────────
@torch.no_grad()
def generate_turn(model, tokenizer, messages, max_new, temp, device):
    """Generate one turn. Returns text only — logprobs computed during GRPO backward.
    
    Key fix: NO return_dict_in_generate, NO output_scores.
    The old code stored scores tensors (~55MB each) which leaked GPU memory
    and caused silent OOM crashes on A100 with dual-role training.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    # Simple generation — no score storage, with repetition penalty to prevent loops
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=max(temp, 0.01),  # Avoid div-by-zero
        top_p=1.0,
        repetition_penalty=1.1,  # Prevent degenerate repeat loops
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return gen_text

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

def run_dual_episode(policy_model, tokenizer, product, device):
    """Run one negotiation with shared policy playing both roles.
    
    The same model plays buyer and seller — role conditioning happens
    via the different system prompts (buyer vs seller).
    This is the SPIRAL self-play approach.
    """
    buyer_prompt = build_buyer_prompt(product)
    seller_prompt_base = build_seller_prompt(product)
    
    buyer_history = []   # Messages for buyer context
    seller_history = []  # Messages for seller context
    all_turns = []       # (role, text) for GRPO
    
    buyer_texts = []     # Only buyer texts (for building seller prompts)
    seller_texts = []    # Only seller texts
    
    final_price = None
    outcome = "TIMEOUT"
    
    # Who goes first? In negotiation, buyer typically starts
    # but for robustness we should also train seller-going-first
    # For now: buyer always starts (matching paper)
    
    for turn in range(MAX_TURNS):
        # ── Buyer turn ──
        msgs = buyer_prompt + buyer_history
        b_text = generate_turn(policy_model, tokenizer, msgs, MAX_NEW_TOKENS, BUYER_TEMP, device)
        b_act = extract_action(b_text)
        
        buyer_history.append({"role": "assistant", "content": b_text})
        buyer_texts.append(b_text)
        all_turns.append(("buyer", b_text))
        
        # Check terminal buyer actions
        if b_act["type"] == "QUIT":
            outcome = "BUYER_QUIT"
            break
        if b_act["type"] == "UNKNOWN":
            outcome = "BUYER_FORMAT_ERROR"
            break
        if b_act["type"] == "DEAL":
            if not seller_texts:
                outcome = "BUYER_DEAL_NO_SELLER_OFFER"
                break
            last_s = extract_action(seller_texts[-1])
            final_price = last_s.get("price")
            outcome = "DEAL_BUYER_ACCEPTS"
            break
        if b_act["type"] == "BUY":
            b_price = b_act["price"]
            if b_price is not None and b_price > product["budget"]:
                outcome = "BUYER_BUDGET_VIOLATION"
                break
        
        # ── Seller turn ──
        # Build seller context from scratch each turn
        s_msgs = build_seller_prompt(product)
        # Add conversation: buyer offers (user) and seller responses (assistant)
        for i, bt in enumerate(buyer_texts):
            s_msgs.append({"role": "user", "content": bt})
            if i < len(seller_texts):
                s_msgs.append({"role": "assistant", "content": seller_texts[i]})
        
        s_text = generate_turn(policy_model, tokenizer, s_msgs, MAX_NEW_TOKENS, SELLER_TEMP, device)
        s_act = extract_action(s_text)
        
        seller_texts.append(s_text)
        buyer_history.append({"role": "user", "content": s_text})
        all_turns.append(("seller", s_text))
        
        # Regulate seller (safety net during early training)
        b_price_val = b_act.get("price")
        r_price, done, reason = regulate_seller(s_act, b_price_val, product)
        
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
    
    buyer_r = compute_buyer_reward(final_price, product["budget"], product["cost"], outcome)
    seller_r = compute_seller_reward(final_price, product["budget"], product["cost"], outcome)
    
    return DualEpisode(
        product=product,
        turns=all_turns,
        final_price=final_price,
        buyer_reward=buyer_r,
        seller_reward=seller_r,
        num_turns=len(all_turns),
        outcome=outcome,
    )


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


# ─── Dual-Role GRPO Update ─────────────────────────────────────────────────────
def dual_role_grpo_update(policy_model, ref_model, tokenizer, episodes,
                           optimizer, rae, device):
    """
    Dual-role GRPO with RAE.

    Optimisations vs first draft (no quality loss):
    1.  _token_logprobs: gather-then-normalise instead of full log_softmax
        → saves ~1 GB VRAM per forward pass
    2.  Single tokenisation: prompt_len computed from the same token ids
        → removes a redundant tokenizer() call per turn
    3.  torch.no_grad for ref model → safe with shared input tensors
    4.  Log-ratio clamped to [-5, 5] before exp → prevents inf ratios
        when policy drifts from ref (main cause of the loss explosion)
    5.  Advantages normalised (zero mean, unit std) per group+role
    """
    policy_model.train()
    G = GROUP_SIZE
    num_groups = len(episodes) // G
    total_loss = 0.0
    turn_count = 0

    for g in range(num_groups):
        group_eps = episodes[g * G : (g + 1) * G]

        # ── RAE advantages per role ──────────────────────────────────
        buyer_advs = torch.tensor(
            [rae.buyer_advantage(ep.buyer_reward) for ep in group_eps],
            dtype=torch.float32, device=device,
        )
        seller_advs = torch.tensor(
            [rae.seller_advantage(ep.seller_reward) for ep in group_eps],
            dtype=torch.float32, device=device,
        )
        # Normalise per role (zero-mean, unit-std) — standard GRPO trick
        def _norm(t):
            if t.numel() < 2:
                return t
            return (t - t.mean()) / (t.std() + 1e-8)
        buyer_advs = _norm(buyer_advs)
        seller_advs = _norm(seller_advs)

        # Update RAE baselines AFTER computing advantages for this group
        for ep in group_eps:
            rae.update(ep.buyer_reward, ep.seller_reward)

        for i, ep in enumerate(group_eps):
            train_seller = random.random() < DUAL_ROLE_RATIO

            for turn_idx, (role, text) in enumerate(ep.turns):
                if role == "seller" and not train_seller:
                    continue

                # ── Build & tokenise prompt + completion in one call ──
                prompt_msgs = _build_turn_prompt(ep, turn_idx)
                prompt_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                full_text = prompt_text + text

                # Tokenise both; derive prompt_len from shared prefix
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
                pol_lp = _token_logprobs(policy_model, ids, attn)  # [1, T-1]

                # ── Reference log-probs (no grad) ──
                with torch.no_grad():
                    ref_lp = _token_logprobs(ref_model, ids, attn)

                # ── Completion-only mask ──
                mask = attn[:, 1:].clone()
                mask[:, :prompt_len - 1] = 0

                # ── Advantage for this role ──
                adv = buyer_advs[i] if role == "buyer" else seller_advs[i]

                # ── Clipped surrogate objective ──
                # Clamp log-ratio to [-5,5] before exp → ratio in [0.007, 148]
                # Prevents the inf / NaN explosions we saw earlier
                log_ratio = (pol_lp - ref_lp).clamp(-5.0, 5.0)
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)

                surr1 = ratio * adv
                surr2 = clipped * adv
                policy_loss = -torch.min(surr1, surr2)

                if KL_COEF > 0:
                    policy_loss = policy_loss + KL_COEF * log_ratio

                loss = (policy_loss * mask).sum() / (mask.sum() + 1e-8)

                # Per-turn backward — accumulates grads, frees graph
                loss.backward()
                total_loss += loss.item()
                turn_count += 1

        # Step after FULL GROUP
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
      then alternating assistant=buyer_text / user=seller_text
    - Seller sees: system=SELLER_SYSTEM, user=setup+inventory+cost,
      then alternating user=buyer_text / assistant=seller_text

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
                prompt_msgs.append({"role": "assistant", "content": prev_text})
            else:  # seller
                prompt_msgs.append({"role": "user", "content": prev_text})
    else:
        prompt_msgs = build_seller_prompt(product)
        for j in range(turn_idx):
            prev_role, prev_text = ep.turns[j]
            if prev_role == "buyer":
                prompt_msgs.append({"role": "user", "content": prev_text})
            else:  # seller
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
    print("=" * 60, flush=True)
    
    # 0. Trackio monitoring
    try:
        import trackio
        run_name = RUN_NAME or f"toy3-{MODEL_NAME.split('/')[-1]}-{NUM_ITERS}it"
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
    
    # 4. Reference model (frozen — for KL/computation of old policy)
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
    
    # 5. Optimizer + RAE
    print(f"\n[5/5] Optimizer (AdamW, lr={LR}) + RAE (decay={RAE_DECAY})...")
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)
    rae = RAE(decay=RAE_DECAY)
    n_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")
    
    # Training
    print(f"\n{'=' * 60}")
    print("DUAL-ROLE GRPO TRAINING WITH RAE")
    print(f"{'=' * 60}")
    
    metrics = []
    t0 = time.time()
    
    for iteration in range(NUM_ITERS):
        t1 = time.time()
        print(f"\n--- Iteration {iteration} ---")
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        products = random.sample(train_products, min(BATCH_SIZE, len(train_products)))
        print(f"  Sampling {len(products)} products, {GROUP_SIZE} rollouts each...")
        
        # Rollout phase
        policy_model.eval()
        episodes = []
        rollout_t0 = time.time()
        for pi, p in enumerate(products):
            for gi in range(GROUP_SIZE):
                ep = run_dual_episode(policy_model, tokenizer, p, dev)
                episodes.append(ep)
            # Progress every 4 products
            if (pi + 1) % 4 == 0 or pi == len(products) - 1:
                elapsed_r = time.time() - rollout_t0
                print(f"  Rollout: {pi+1}/{len(products)} products, "
                      f"{len(episodes)} episodes, {elapsed_r:.0f}s", flush=True)
        
        # Clear GPU cache after rollout
        torch.cuda.empty_cache()
        gc.collect()
        
        # GRPO update
        policy_model.train()
        print(f"  Dual-role GRPO update on {len(episodes)} episodes...")
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
        
        elapsed = time.time() - t1
        print(f"  Loss={loss:.4f} BuyerR={mean_br:.4f} SellerR={mean_sr:.4f} "
              f"Deal={deal_rate:.1%} Price=${mean_price:.2f} Turns={mean_turns:.1f} "
              f"Time={elapsed:.1f}s")
        print(f"  Outcomes: {dict(sorted(outcomes.items(), key=lambda x: -x[1])[:5])}")
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
                    "perf/vram_gb": torch.cuda.memory_allocated()/1e9,
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
            "rae_state": rae.state_dict(),
            "outcomes": outcomes,
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

