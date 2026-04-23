"""Standalone test for negotiation environment + model generation.
No TRL yet — just verify the pipeline components."""
import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_loader import load_dataset, format_inventory, format_shopping_list, format_seller_private


BUYER_SYSTEM_PROMPT = """You are a buyer looking forward to buying things on your Shopping List from me, the seller.
You have access to the seller's Inventory List and you can bargain about the prices.
Your task is to bargain with the seller and reach a deal with the price as low as possible in limited turns.
You can only buy things on the Shopping List in the limited quantity. Use the codename of the product, instead of the title.
You can only buy things that cost less than your budget, otherwise, you should quit negotiating.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the seller. Speak concisely and cut to the chase.
Action: one of the limited actions that define the real intention of your Talk.
1. '[BUY] $M (N codename_1)' if you wish to offer the seller $M to purchase all N items of the product with the codename "codename_1".
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the seller.
3. '[DEAL] $M (N codename_1)' if you finally accept on a former offer proposed by the seller. $M (N codename_1) is an exact copy of seller's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.

Your reply should strictly follow this format, for example:
Thought: I'm a buyer and I want to bargain. The listing price is $15, which is too expensive, so I try to buy for $10.
Talk: Hello, I'm tight on budget. can you sell it for 10$?
Action: [BUY] $10 (1x apple_1)
"""


SELLER_SYSTEM_PROMPT = """You are a seller looking forward to selling things on your Inventory List to me, the buyer.
Your task is to bargain with the buyer and reach a deal with the price as high as possible in limited turns.
You can only sell things that are on the Inventory List. Use the codename of the product, instead of the title.
You have access to private information: the cost price of each product in the Inventory List, and do not disclose the real cost to the buyer.
You should only agree on a deal when the selling price is higher than the cost, otherwise, you should quit negotiating.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the buyer. Speak concisely and cut to the chase;
Action: one of the limited actions that define the real intention of your Talk.
1. '[SELL] $M (N codename_1)' if you want to propose selling N items of the product with the codename "codename_1" to the buyer for the total price of $M.
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the buyer.
3. '[DEAL] $M (N codename_1)' if you finally agree on a former offer proposed by the buyer, and sell N items of the product with the codename "codename_1" to the buyer for the total price of $M. $M (N codename_1) is an exact copy of buyer's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
"""


def build_buyer_prompt(product: dict) -> list:
    """Build the initial conversation for a buyer."""
    inv = format_inventory(product)
    need = format_shopping_list(product)
    user_msg = f"{inv}\n\n{need}\n\nNow, I play the role of seller and you play the role of buyer. We are going to negotiate based on the Inventory List in 6 turns."
    return [
        {"role": "system", "content": BUYER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def build_seller_prompt(product: dict) -> list:
    """Build the initial conversation for a seller."""
    inv = format_inventory(product)
    private = format_seller_private(product)
    user_msg = f"{inv}\n\n{private}\n\nNow, I play the role of buyer and you play the role of seller. We are going to negotiate based on the Inventory List in 6 turns."
    return [
        {"role": "system", "content": SELLER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def extract_action(text: str) -> dict:
    """Extract action from model output."""
    # Try to find [BUY], [SELL], [DEAL], [REJECT], [QUIT]
    match = re.search(r'\[(BUY|SELL|DEAL|REJECT|QUIT)\](?:\s*\$([\d.,]+))?(?:\s*\(([^)]*)\))?', text, re.IGNORECASE)
    if match:
        action_type = match.group(1).upper()
        money = match.group(2)
        objects = match.group(3)
        return {
            "type": action_type,
            "money": float(money.replace(",", "")) if money else None,
            "objects": objects.strip() if objects else None,
            "raw": match.group(0),
        }
    return {"type": "UNKNOWN", "money": None, "objects": None, "raw": text[-100:]}


def compute_reward_buyer(action: dict, budget: float, cost: float) -> float:
    """Compute buyer reward per paper formula."""
    action_type = action.get("type", "UNKNOWN")
    
    # Format violation
    if action_type == "UNKNOWN":
        return -1.0
    
    # Budget violation
    money = action.get("money", 0) or 0
    if money > budget:
        return -1.0
    
    # Terminal actions
    if action_type == "QUIT":
        return 0.0
    
    if action_type == "DEAL":
        # MI scenario: budget > cost
        if budget > cost:
            # Normalize: (budget - price) / |budget - cost|
            reward = (budget - money) / abs(budget - cost)
            return max(-1.0, min(1.0, reward))
        else:
            # CI scenario: budget < cost — deal is bad
            reward = (budget - money) / abs(budget - cost)
            return max(-1.0, min(1.0, reward))
    
    # Non-terminal: [BUY], [REJECT] — no reward yet (episode continues)
    return None  # None = no terminal reward


def simple_seller_response(buyer_action: dict, product: dict) -> str:
    """Simple rule-based seller for testing."""
    cost = product["cost"]
    list_price = product["list_price"]
    budget = product["budget"]
    codename = product["codename"]
    
    action_type = buyer_action.get("type", "")
    money = buyer_action.get("money", 0) or 0
    
    if action_type == "QUIT":
        return "The buyer has quit the negotiation."
    
    if action_type == "DEAL":
        if money >= cost:
            return "Deal accepted!"
        else:
            return "I cannot sell below my cost. Deal rejected."
    
    if action_type == "BUY":
        if money >= list_price * 0.9:
            # Close to list price — accept
            return f"Thought: The buyer is offering close to list price.\nTalk: That works for me!\nAction: [DEAL] ${money:.0f} (1x {codename})"
        elif money >= cost * 1.3:
            # Reasonable offer — counter slightly higher
            counter = min(money * 1.15, list_price * 0.95)
            return f"Thought: I can go a bit higher than the buyer's offer.\nTalk: How about ${counter:.0f}?\nAction: [SELL] ${counter:.0f} (1x {codename})"
        elif money >= cost:
            # Low but above cost — counter significantly
            counter = (money + list_price) / 2
            return f"Thought: The offer is low but above cost. I'll counter higher.\nTalk: That's too low. I can offer ${counter:.0f}.\nAction: [SELL] ${counter:.0f} (1x {codename})"
        else:
            # Below cost — reject
            return f"Thought: The offer is below my cost.\nTalk: I can't go that low. My best offer is ${list_price * 0.9:.0f}.\nAction: [SELL] ${list_price * 0.9:.0f} (1x {codename})"
    
    if action_type == "REJECT":
        return f"Thought: The buyer rejected. I'll propose my price.\nTalk: I'm offering ${list_price * 0.95:.0f} for this item.\nAction: [SELL] ${list_price * 0.95:.0f} (1x {codename})"
    
    return "Thought: I'm not sure what to do.\nTalk: Let's continue.\nAction: [REJECT]"


def run_single_negotiation(model, tokenizer, product: dict, max_turns: int = 6, device: str = "cuda") -> dict:
    """Run a single negotiation episode. Returns episode data."""
    messages = build_buyer_prompt(product)
    budget = product["budget"]
    cost = product["cost"]
    
    episode_log = []
    final_price = None
    action_type = None
    
    for turn in range(max_turns):
        # Generate buyer response
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        action = extract_action(response_text)
        action_type = action["type"]
        
        episode_log.append({
            "turn": turn,
            "role": "buyer",
            "text": response_text,
            "action": action,
        })
        
        # Check terminal
        if action_type in ("DEAL", "QUIT"):
            if action_type == "DEAL":
                final_price = action.get("money", 0)
            break
        
        # Get seller response
        seller_text = simple_seller_response(action, product)
        seller_action = extract_action(seller_text)
        
        episode_log.append({
            "turn": turn,
            "role": "seller",
            "text": seller_text,
            "action": seller_action,
        })
        
        # Check if seller accepted
        if seller_action["type"] == "DEAL":
            final_price = seller_action.get("money", 0)
            # End with buyer accepting
            break
        
        # Add to messages
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": seller_text})
    
    # Compute reward
    if final_price is not None:
        reward = compute_reward_buyer({"type": "DEAL", "money": final_price}, budget, cost)
    else:
        reward = compute_reward_buyer({"type": "QUIT"}, budget, cost)
    
    return {
        "product": product,
        "log": episode_log,
        "final_price": final_price,
        "reward": reward,
        "turns": len([e for e in episode_log if e["role"] == "buyer"]),
    }


def main():
    print("Loading dataset...")
    ds = load_dataset()
    
    # Pick a small model for quick testing
    model_name = "Qwen/Qwen3-1.7B"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    
    # Run 3 test negotiations
    test_products = ds["test"][:3]
    for product in test_products:
        print(f"\n{'='*60}")
        print(f"Product: {product['title'][:60]}...")
        print(f"Budget: ${product['budget']:.2f}, Cost: ${product['cost']:.2f}, MI: {product['mi']}")
        
        result = run_single_negotiation(model, tokenizer, product, max_turns=6, device=device)
        
        for entry in result["log"]:
            role = entry["role"].upper()
            action = entry["action"]
            print(f"  [{role}] {action['type']} money=${action.get('money')}")
        
        print(f"  REWARD: {result['reward']:.4f} | Price: ${result['final_price']}")


if __name__ == "__main__":
    main()
