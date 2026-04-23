"""
Toy Run 1: Minimal GRPO for negotiation (single-turn buyer, rule seller).
Goal: Validate pipeline: model loading, generation, reward computation, update.
Uses Qwen3-1.7B on a small GPU with LoRA to prevent OOM.
"""
import os
import sys
import re
import math
import json
import random
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from data_loader import load_dataset, format_inventory, format_shopping_list

# ─── Diagnostics ──────────────────────────────────────────────────────────────
def log_cuda():
    print(f"torch.__version__: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        x = torch.randn(2, 2).cuda()
        y = x @ x.T
        print(f"CUDA tensor op OK: {y.device}")
    else:
        print("WARNING: No CUDA available — will fail!")
        sys.exit(1)


# ─── Prompts ─────────────────────────────────────────────────────────────────
BUYER_SYSTEM_PROMPT = """You are a buyer negotiating for the best possible price.

Format your reply EXACTLY as:
Thought: <your strategic reasoning>
Talk: <what you say to the seller>
Action: [BUY] $<price> (1x <codename>)

You must include all three parts. The Action must be one of:
- [BUY] $M (1x codename)  — make an offer
- [DEAL] $M (1x codename) — accept seller's offer
- [REJECT] — reject and wait
- [QUIT] — give up and end negotiation"""


def build_prompt(product: dict) -> list:
    inv = format_inventory(product)
    need = format_shopping_list(product)
    user = (
        f"{inv}\n\n{need}\n\n"
        f"Now, I play the role of seller and you play the role of buyer. "
        f"We are going to negotiate based on the Inventory List in 6 turns. "
        f"Make your first offer now."
    )
    return [
        {"role": "system", "content": BUYER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


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
        return {"type": m.group(1).upper(), "price": price, "raw": m.group(0)}
    return {"type": "UNKNOWN", "price": None, "raw": text[-120:]}


# ─── Simple Rule Seller ──────────────────────────────────────────────────────
def simple_seller_responds(buyer_action: dict, product: dict) -> tuple:
    cost = product["cost"]
    budget = product["budget"]
    list_price = product["list_price"]
    codename = product["codename"]
    atype = buyer_action["type"]
    price = buyer_action["price"] or 0

    if atype == "QUIT":
        return "The buyer has quit the negotiation.", None, True
    if atype == "UNKNOWN":
        return "Invalid format.", None, True
    if atype == "DEAL":
        if price >= cost:
            return "Deal accepted!", price, True
        return "I cannot sell below my cost.", None, True
    if atype == "BUY":
        if price >= cost * 1.3:
            return f"Thought: Good offer.\nTalk: Works for me!\nAction: [DEAL] ${price:.0f} (1x {codename})", price, True
        elif price >= cost:
            counter = (price + list_price) / 2
            return f"Thought: I can counter.\nTalk: How about ${counter:.0f}?\nAction: [SELL] ${counter:.0f} (1x {codename})", None, False
        else:
            counter = (price + list_price * 0.95) / 2
            return f"Thought: Too low.\nTalk: My best is ${counter:.0f}.\nAction: [SELL] ${counter:.0f} (1x {codename})", None, False
    return "Thought: Waiting.\nTalk: ...\nAction: [REJECT]", None, False


# ─── Reward ──────────────────────────────────────────────────────────────────
def compute_reward(final_price, budget, cost):
    if final_price is None:
        return 0.0  # QUIT / no deal
    denom = abs(budget - cost)
    if denom < 1e-6:
        return 0.0
    reward = (budget - final_price) / denom
    return max(-1.0, min(1.0, reward))


# ─── Generation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_completions(model, tokenizer, prompts, num_generations=4,
                         max_new_tokens=120, temperature=1.0):
    """Generate G completions per prompt."""
    all_results = []
    for prompt in prompts:
        prompt_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                               truncation=True, max_length=1024).to(model.device)
        results = []
        for _ in range(num_generations):
            outputs = model.generate(
                **prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            gen_tokens = outputs.sequences[0][prompt_ids["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # logprob sum
            scores = torch.stack(outputs.scores, dim=1)
            log_probs = F.log_softmax(scores, dim=-1)
            token_lp = torch.gather(log_probs[0], 1, gen_tokens.unsqueeze(-1)).squeeze(-1)

            results.append({
                "text": gen_text,
                "logprob_sum": token_lp.sum().item(),
                "token_log_probs": token_lp,
                "gen_tokens": gen_tokens,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids["input_ids"],
            })
        all_results.append(results)
    return all_results


# ─── GRPO Update ─────────────────────────────────────────────────────────────
def grpo_update(model, ref_model, tokenizer, batch_data, optimizer, epsilon=0.2):
    model.train()
    total_loss = 0.0
    for item in batch_data:
        prompt_text = item["prompt_text"]
        gen_tokens = item["gen_tokens"]
        advantage = item["advantage"]

        full_text = prompt_text + item["text"]
        full_ids = tokenizer(full_text, return_tensors="pt",
                             truncation=True, max_length=2048).to(model.device)

        prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                               truncation=True, max_length=2048).to(model.device)["input_ids"]
        prompt_len = prompt_ids.shape[1]

        # Policy forward
        outputs = model(**full_ids)
        logits = outputs.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_lp = torch.gather(log_probs, 2,
                                full_ids["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)

        # Reference forward
        with torch.no_grad():
            ref_outputs = ref_model(**full_ids)
            ref_logits = ref_outputs.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_token_lp = torch.gather(ref_log_probs, 2,
                                        full_ids["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)

        mask = full_ids["attention_mask"][:, 1:].clone()
        mask[:, :prompt_len - 1] = 0

        ratio = torch.exp(token_lp - ref_token_lp)
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

        surr1 = ratio * advantage
        surr2 = clipped * advantage
        policy_loss = -torch.min(surr1, surr2)

        loss = (policy_loss * mask).sum() / (mask.sum() + 1e-8)
        total_loss += loss

    avg_loss = total_loss / len(batch_data)
    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return avg_loss.item()


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    log_cuda()

    # Dataset
    print("\n[1/6] Loading dataset...")
    ds = load_dataset()
    train_products = ds["train"]
    print(f"Train products: {len(train_products)}")

    # Model + Tokenizer
    model_name = "Qwen/Qwen3-1.7B"
    print(f"\n[2/6] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[3/6] Loading policy model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    print("\n[4/6] Applying LoRA...")
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Reference model (frozen, no LoRA)
    print("\n[5/6] Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # Training config
    num_iterations = 5
    batch_size = 8
    group_size = 4
    max_new_tokens = 100

    print(f"\n[6/6] Starting training: {num_iterations} iters, {batch_size} products, G={group_size}")

    metrics_log = []
    start_time = time.time()

    for iteration in range(num_iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration} ---")

        # Sample products
        products = random.sample(train_products, min(batch_size, len(train_products)))
        prompts = [build_prompt(p) for p in products]

        # Generate
        print("  Generating...")
        completions = generate_completions(
            model, tokenizer, prompts,
            num_generations=group_size, max_new_tokens=max_new_tokens,
        )

        # Evaluate
        rewards_all = []
        for i, product in enumerate(products):
            group_rewards = []
            group_data = []
            for comp in completions[i]:
                action = extract_action(comp["text"])
                seller_text, final_price, done = simple_seller_responds(action, product)
                reward = compute_reward(final_price, product["budget"], product["cost"])
                group_rewards.append(reward)
                group_data.append({**comp, "action": action, "reward": reward,
                                   "final_price": final_price})

            mean_reward = sum(group_rewards) / len(group_rewards)
            for gd in group_data:
                gd["advantage"] = gd["reward"] - mean_reward
            rewards_all.extend(group_data)

        # Update
        print("  Updating...")
        loss = grpo_update(model, ref_model, tokenizer, rewards_all, optimizer)

        # Metrics
        mean_reward = sum(d["reward"] for d in rewards_all) / len(rewards_all)
        mean_adv = sum(d["advantage"] for d in rewards_all) / len(rewards_all)
        deal_rate = sum(1 for d in rewards_all if d["final_price"] is not None) / len(rewards_all)
        mean_price = sum(d["final_price"] for d in rewards_all if d["final_price"] is not None) / max(1, sum(1 for d in rewards_all if d["final_price"] is not None))

        print(f"  Loss: {loss:.4f} | Reward: {mean_reward:.4f} | "
              f"Adv: {mean_adv:.4f} | Deal: {deal_rate:.1%} | "
              f"MeanPrice: ${mean_price:.2f} | "
              f"Time: {time.time() - iter_start:.1f}s")

        metrics_log.append({
            "iteration": iteration,
            "loss": loss,
            "mean_reward": mean_reward,
            "mean_advantage": mean_adv,
            "deal_rate": deal_rate,
            "mean_price": mean_price,
            "time": time.time() - iter_start,
        })

        # Sample
        sample = rewards_all[0]
        print(f"  Sample: {sample['action']['type']} ${sample['action']['price']} "
              f"-> R={sample['reward']:.4f} | text[:80]: {sample['text'][:80]}")

    # Save
    print("\nSaving model...")
    save_dir = "/app/toy1_model"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save metrics
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"\nToy Run 1 complete! Total time: {time.time() - start_time:.1f}s")
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
