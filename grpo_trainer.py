"""Custom GRPO trainer for multi-turn negotiation.
Avoids TRL's experimental environment_factory/tool-calling loop.
Based on the paper's approach (on-policy GRPO with multi-turn episodes).
"""
import math
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Callable, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from accelerate import Accelerator
import trackio


class NegotiationEpisode:
    """One complete negotiation episode."""
    def __init__(self, product: dict, buyer_messages: list, seller_messages: list,
                 final_action: dict, reward: float, turns: int):
        self.product = product
        self.buyer_messages = buyer_messages  # list of buyer response texts
        self.seller_messages = seller_messages  # list of seller response texts
        self.final_action = final_action
        self.reward = reward
        self.turns = turns


class SimpleGRPOTrainer:
    """Simplified GRPO trainer for negotiation.
    
    Key hyperparameters from paper 2604.09855:
    - lr = 3e-5 (30x higher than TRL default!)
    - KL penalty = 0 (no KL divergence regularization)
    - Group size G = 8
    - Scale rewards = None (raw clipping, no normalization)
    """
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        lr: float = 3e-5,
        kl_coef: float = 0.0,
        group_size: int = 8,
        max_completion_length: int = 300,
        temperature: float = 1.0,
        device: str = "cuda",
        use_8bit_adam: bool = False,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.group_size = group_size
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Setup optimizer
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
            except ImportError:
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        self.step_count = 0
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Trackio init
        try:
            trackio.init(
                project="anchor-negotiation",
                name=f"grpo_negotiation_{device}",
            )
        except Exception as e:
            print(f"Trackio init warning: {e}")
    
    def generate_episodes(
        self,
        products: List[dict],
        generate_fn: Callable,
        num_generations: Optional[int] = None,
    ) -> List[NegotiationEpisode]:
        """Generate G episodes per product using the current policy."""
        G = num_generations or self.group_size
        all_episodes = []
        
        for product in products:
            for _ in range(G):
                episode = generate_fn(self.model, self.tokenizer, product, self.device)
                all_episodes.append(episode)
        
        return all_episodes
    
    def compute_logprobs(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for each token in the sequence."""
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits  # [batch, seq_len, vocab]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather logprobs for actual tokens
        # log_probs[token_ids] for each position
        token_log_probs = torch.gather(log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        return token_log_probs
    
    def compute_grpo_loss(
        self,
        episodes: List[NegotiationEpisode],
        prompt_texts: List[str],
    ) -> tuple:
        """Compute GRPO loss for a batch of episodes.
        
        GRPO objective (per group):
        1. Compute mean reward for the group: r̄
        2. Compute advantage: A = r - r̄
        3. Compute policy logprobs π_θ(y|x) for each completion
        4. Compute reference logprobs π_ref(y|x)
        5. Loss = -mean[ (π_θ / π_ref) * A ]  (with clipping)
        
        Since we have no KL penalty (paper uses KL=0), we skip the KL term.
        """
        self.model.train()
        
        # Group episodes by product
        # episodes are ordered: product_0_gen_0, product_0_gen_1, ..., product_1_gen_0, ...
        G = self.group_size
        num_products = len(episodes) // G
        
        losses = []
        metrics = {
            "loss": 0.0,
            "mean_reward": 0.0,
            "mean_advantage": 0.0,
            "mean_ratio": 0.0,
        }
        
        for i in range(num_products):
            group_eps = episodes[i * G : (i + 1) * G]
            rewards = torch.tensor([ep.reward for ep in group_eps], dtype=torch.float32, device=self.device)
            
            # Advantage = reward - group_mean
            mean_reward = rewards.mean()
            advantages = rewards - mean_reward
            
            # Normalize advantages (optional — paper doesn't mention this)
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            group_losses = []
            group_ratios = []
            
            for j, ep in enumerate(group_eps):
                prompt_text = prompt_texts[i]
                completion_text = "\n".join(ep.buyer_messages)
                full_text = prompt_text + completion_text
                
                # Tokenize
                tokens = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.device)
                
                input_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]
                
                # Get prompt length
                prompt_tokens = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.device)["input_ids"]
                prompt_len = prompt_tokens.shape[1]
                
                # Forward pass through policy model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Logprobs for policy
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                token_log_probs = torch.gather(
                    log_probs, 2, input_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                
                # Mask: only completion tokens (not prompt tokens)
                completion_mask = attention_mask[:, 1:].clone()
                completion_mask[:, :prompt_len - 1] = 0  # Zero out prompt positions
                
                # Compute ref logprobs
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
                    ref_logits = ref_outputs.logits
                    ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                    ref_token_log_probs = torch.gather(
                        ref_log_probs, 2, input_ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)
                
                # Ratio for each token
                ratio = torch.exp(token_log_probs - ref_token_log_probs)
                
                # Clipped ratio (PPO-style clipping)
                clipped_ratio = torch.clamp(ratio, 0.2, 1.8)  # ε = 0.2 typical
                
                # GRPO loss: -mean[min(ratio, clipped_ratio) * advantage]
                # For advantage > 0: use min; for advantage < 0: use max (but with clipping it's symmetric)
                surrogate1 = ratio * advantages[j]
                surrogate2 = clipped_ratio * advantages[j]
                policy_loss = -torch.min(surrogate1, surrogate2)
                
                # Average over completion tokens
                masked_loss = (policy_loss * completion_mask).sum() / (completion_mask.sum() + 1e-8)
                
                group_losses.append(masked_loss)
                group_ratios.append(ratio.mean().item())
            
            # Group loss
            group_loss = torch.stack(group_losses).mean()
            losses.append(group_loss)
            
            metrics["mean_reward"] += mean_reward.item()
            metrics["mean_advantage"] += advantages.mean().item()
            metrics["mean_ratio"] += sum(group_ratios) / len(group_ratios)
        
        # Average over all groups
        total_loss = torch.stack(losses).mean()
        
        metrics["loss"] = total_loss.item()
        metrics["mean_reward"] /= num_products
        metrics["mean_advantage"] /= num_products
        metrics["mean_ratio"] /= num_products
        
        return total_loss, metrics
    
    def train_step(self, episodes: List[NegotiationEpisode], prompt_texts: List[str]) -> Dict[str, float]:
        """Single GRPO training step."""
        self.optimizer.zero_grad()
        
        # Compute loss
        loss, metrics = self.compute_grpo_loss(episodes, prompt_texts)
        
        # Backward
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        self.step_count += 1
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log metrics to console and trackio."""
        print(f"Iter {iteration}: loss={metrics['loss']:.4f} "
              f"reward={metrics['mean_reward']:.4f} "
              f"adv={metrics['mean_advantage']:.4f} "
              f"ratio={metrics['mean_ratio']:.4f}")
        
        try:
            trackio.log(metrics, step=iteration)
        except Exception:
            pass
