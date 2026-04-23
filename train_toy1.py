"""Toy Run 1: Minimal GRPO for negotiation (single-turn buyer, rule seller).
Goal: Validate pipeline: model loading, generation, reward computation, update.
Uses Qwen3-1.7B on T4-small.
"""
import os
import re
import math
import json
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_loader import load_dataset, format_inventory, format_shopping_list


def parse_price(price_str: str) -> float:
    return float(price_str.replace("$", "").replace(",", "").strip())


BUYER_SYSTEM_PROMPT = """You are a buyer negotiating for the best price.

Format your reply as:
Thought: <your strategic thinking>
Talk: <what you say to the seller>
Action: [BUY] $<price> (1x <codename>)"""


def build_prompt(product: dict) -> str:
    inv = format_inventory(product)
    need = format_shopping_list(product)
    user = f"{inv}\n\n{need}\n\nNegotiate in 6 turns. Make your first offer now."
    messages = [
        {"role": "system", "content": BUYER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return messages


def extract_action(text: str) -> dict:
    """Extract [BUY], [DEAL], [REJECT], [QUIT] and price."""
    # Look for [BUY] $X or [BUY] $X.XX
    m = re.search(r'\[(BUY|SELL|DEAL|REJECT|QUIT)\](?:\s*\$([\d,.]+))?(?:\s*\(([^)]*)\))?', text, re.I)
    if m:
        price_str = m.group(2)
        price = float(price_str.replace(",", "")) if price_str else None
        return {"type": m.group(1).upper(), "price": price, "raw": m.group(0)}
    return {"type": "UNKNOWN", "price": None, "raw": text[-80:]}


def simple_seller_responds(buyer_action: dict, product: dict) -> tuple:
    """Rule-based seller. Returns (seller_text, final_price, done)."""
    cost = product["cost"]
    budget = product["budget"]
    list_price = product["list_price"]
    codename = product["codename"]
    
    atype = buyer_action["type"]
    price = buyer_action["price"] or 0
    
    if atype == "QUIT":
        return "The buyer quit.", None, True
    
    if atype == "UNKNOWN":
        return "Invalid format.", None, True
    
    if atype == "BUY":
        if price >= cost * 1.3:
            # Good offer — accept
            return f"[SELL] ${price:.0f} (1x {codename})", price, True
        elif price >= cost:
            # Reasonable — counter
            counter = (price + list_price) / 2
            return f"[SELL] ${counter:.0f} (1x {codename})", None, False
        else:
            # Too low
            counter = (price + list_price * 0.95) / 2
            return f"[SELL] ${counter:.0f} (1x {codename})", None, False
    
    return "[REJECT]", None, False


def compute_reward(final_price: float, budget: float, cost: float) -> float:
    """Paper's reward formula, simplified for single-turn acceptance."""
    if final_price is None:
        return 0.0  # QUIT / no deal
    
    if budget > cost:
        # MI scenario
        reward = (budget - final_price) / abs(budget - cost)
    else:
        # CI scenario
        reward = (budget - final_price) / abs(budget - cost)
    
    return max(-1.0, min(1.0, reward))


def generate_completions(model, tokenizer, prompts: list, num_generations: int = 4,
                         max_new_tokens: int = 150, temperature: float = 1.0, device: str = "cuda") -> list:
    """Generate G completions for each prompt. Returns list of (text, logprob_sum) tuples per prompt."""
    all_results = []
    
    for prompt in prompts:
        # Convert to text
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        results = []
        for _ in range(num_generations):
            with torch.no_grad():
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
            
            # Compute logprob sum for generated tokens
            scores = torch.stack(outputs.scores, dim=1)  # [1, gen_len, vocab]
            log_probs = F.log_softmax(scores, dim=-1)
            token_log_probs = torch.gather(log_probs[0], 1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            logprob_sum = token_log_probs.sum().item()
            
            results.append({
                "text": gen_text,
                "logprob_sum": logprob_sum,
                "token_log_probs": token_log_probs,
                "gen_tokens": gen_tokens,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids["input_ids"],
            })
        
        all_results.append(results)
    
    return all_results


def grpo_update(model, ref_model, tokenizer, batch_data: list, optimizer, device: str, epsilon: float = 0.2):
    """GRPO update for a batch of (prompt, completion, advantage) triples.
    
    batch_data: list of dicts with keys:
        - prompt_text: str
        - gen_tokens: tensor of completion token IDs
        - advantages: scalar advantage value
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    for item in batch_data:
        prompt_text = item["prompt_text"]
        gen_tokens = item["gen_tokens"]
        advantage = item["advantage"]
        
        # Full text
        full_text = prompt_text + item["text"]
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        
        # Prompt length
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(device)["input_ids"]
        prompt_len = prompt_ids.shape[1]
        
        # Forward pass policy
        outputs = model(**full_ids)
        logits = outputs.logits
        
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = torch.gather(log_probs, 2, full_ids["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        
        # Forward pass reference
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
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        
        # GRPO surrogate loss
        surr1 = ratio * advantage
        surr2 = clipped_ratio * advantage
        policy_loss = -torch.min(surr1, surr2)
        
        # Masked mean
        loss = (policy_loss * mask).sum() / (mask.sum() + 1e-8)
        
        total_loss += loss
        total_tokens += mask.sum().item()
    
    # Average
    avg_loss = total_loss / len(batch_data)
    
    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return avg_loss.item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load dataset
    ds = load_dataset()
    train_products = ds["train"]
    
    # Load model
    model_name = "Qwen/Qwen3-1.7B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    print(f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    
    # Training loop
    num_iterations = 5
    batch_size = 8  # products per iteration
    group_size = 4   # generations per product
    
    for iteration in range(num_iterations):
        # Sample products
        products = random.sample(train_products, min(batch_size, len(train_products)))
        prompts = [build_prompt(p) for p in products]
        
        # Generate completions
        print(f"\nIter {iteration}: generating {len(products)} x {group_size} completions...")
        completions = generate_completions(model, tokenizer, prompts, num_generations=group_size, device=device)
        
        # Evaluate each completion
        rewards_all = []
        for i, product in enumerate(products):
            group_rewards = []
            group_data = []
            for comp in completions[i]:
                action = extract_action(comp["text"])
                seller_text, final_price, done = simple_seller_responds(action, product)
                reward = compute_reward(final_price, product["budget"], product["cost"])
                group_rewards.append(reward)
                group_data.append({
                    **comp,
                    "action": action,
                    "reward": reward,
                    "final_price": final_price,
                })
            
            # Compute advantages
            mean_reward = sum(group_rewards) / len(group_rewards)
            for j, gd in enumerate(group_data):
                gd["advantage"] = gd["reward"] - mean_reward
            
            rewards_all.extend(group_data)
        
        # GRPO update
        loss = grpo_update(model, ref_model, tokenizer, rewards_all, optimizer, device)
        
        # Metrics
        mean_reward = sum(d["reward"] for d in rewards_all) / len(rewards_all)
        mean_advantage = sum(d["advantage"] for d in rewards_all) / len(rewards_all)
        deal_rate = sum(1 for d in rewards_all if d["final_price"] is not None) / len(rewards_all)
        
        print(f"  Loss: {loss:.4f} | Mean Reward: {mean_reward:.4f} | "
              f"Adv: {mean_advantage:.4f} | Deal Rate: {deal_rate:.2%}")
        
        # Sample one completion
        sample = rewards_all[0]
        print(f"  Sample: {sample['action']['type']} ${sample['action']['price']} "
              f"-> Reward: {sample['reward']:.4f}")
    
    print("\nToy Run 1 complete!")
    
    # Save
    save_path = "/app/anchor_negotiation/toy1_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
