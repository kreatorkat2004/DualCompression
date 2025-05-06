import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from bert_score import score as bert_score

class DynamicPromptCompressionAgent(nn.Module):
    """
    Implementation of the Dynamic Prompt Compression (DPC) agent from Section 3.4
    
    The DPC agent sequentially decides which tokens to keep and which to remove
    from a prompt, formulating compression as a Markov Decision Process (MDP).
    """
    
    def __init__(
        self,
        encoder_model_name="xlm-roberta-large",
        device=None,
        max_length=512
    ):
        super().__init__()
        
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        #Load XLM-RoBERTa-Large from paper
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.encoder = AutoModel.from_pretrained(encoder_model_name).to(self.device)
        
        self.max_length = max_length
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  #binary classification
        ).to(self.device)
        
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
    
    def forward(self, prompt_ids):
        """Forward pass through the actor-critic network"""
        with torch.no_grad():
            attention_mask = (prompt_ids != self.tokenizer.pad_token_id).to(self.device)
            outputs = self.encoder(
                input_ids=prompt_ids,
                attention_mask=attention_mask
            )
            embeddings = outputs.last_hidden_state
        
        policy_logits = self.policy_head(embeddings)
        
        value = self.value_head(embeddings[:, 0, :])
        
        return policy_logits, value
    
    def get_action_and_value(self, prompt_ids, temperature=1.0):
        """Get action (keep/drop decisions) and value for a given prompt"""
        policy_logits, value = self.forward(prompt_ids)
        
        scaled_logits = policy_logits / temperature
        
        probs = F.softmax(scaled_logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        action_log_probs = torch.gather(log_probs, -1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, action_log_probs, value
    
    def apply_mask(self, prompt_ids, action):
        """Apply the action mask to the prompt"""
        prompt_np = prompt_ids[0].cpu().numpy()
        action_np = action[0].cpu().numpy()
        
        kept_indices = np.where(action_np == 1)[0]
        kept_tokens = prompt_np[kept_indices]
        
        if len(kept_tokens) == 0:
            kept_tokens = np.array([prompt_np[0]])
        
        new_prompt_ids = torch.tensor(kept_tokens).unsqueeze(0).to(self.device)
        
        return new_prompt_ids
    
    def compress_prompt(self, prompt, target_ratio=0.5, max_steps=3, temperature=0.5):
        """
        Compress a prompt by sequentially removing tokens
        
        Args:
            prompt: The text prompt to compress
            target_ratio: Target compression ratio (0-1, lower means more compression)
            max_steps: Maximum number of compression iterations
            temperature: Sampling temperature (higher = more randomness)
            
        Returns:
            Compressed prompt text
        """
        encoded = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)
        
        current_ids = encoded.input_ids
        original_length = current_ids.size(1)
        
        for step in range(max_steps):
            action, _, _ = self.get_action_and_value(current_ids, temperature)
            
            new_ids = self.apply_mask(current_ids, action)
            
            current_ratio = new_ids.size(1) / original_length
            
            if current_ratio <= target_ratio or new_ids.size(1) == current_ids.size(1):
                break
            
            current_ids = new_ids
        
        compressed_prompt = self.tokenizer.decode(new_ids[0], skip_special_tokens=True)
        
        return compressed_prompt
    
    def compute_reward(
        self, 
        original_prompt, 
        compressed_prompt, 
        original_output=None,
        compressed_output=None,
        alpha=0.5,   #Compression reward weight
        beta=0.3,    #Semantic similarity weight
        gamma=0.2,   #Output distribution similarity weight
        target_ratio=0.5
    ):
        """
        Compute reward for prompt compression as defined in Section 3.4
        
        R(s_t,a_t) = α·ρ^(-1) + β·D(s_0,s_t) - γ·KL(P(y|s_t),P(y|s_0))
                      - 𝟙_{ρ<c_s}·P_s - 𝟙_{ρ>c_l}·P_l
        
        Args:
            original_prompt: Original uncompressed prompt
            compressed_prompt: Compressed prompt after applying mask
            original_output: Model output with original prompt (optional)
            compressed_output: Model output with compressed prompt (optional)
            alpha: Weight for compression ratio reward
            beta: Weight for semantic preservation reward
            gamma: Weight for output distribution preservation reward
            target_ratio: Target compression ratio
            
        Returns:
            Reward score
        """
        orig_enc = self.tokenizer(original_prompt, return_tensors="pt").to(self.device)
        comp_enc = self.tokenizer(compressed_prompt, return_tensors="pt").to(self.device)
        
        rho = comp_enc.input_ids.size(1) / orig_enc.input_ids.size(1)
        
        compression_reward = alpha * (1.0 / rho)
        
        #semantic similarity reward: β·D(s_0,s_t)
        with torch.no_grad():
            orig_output = self.encoder(**orig_enc)
            comp_output = self.encoder(**comp_enc)
            
            orig_emb = orig_output.last_hidden_state[:, 0, :]
            comp_emb = comp_output.last_hidden_state[:, 0, :]
            
            similarity = F.cosine_similarity(orig_emb, comp_emb).item()
            semantic_reward = beta * similarity
        
        #output distribution similarity penalty: -γ·KL(P(y|s_t),P(y|s_0))
        kl_divergence = 0.0
        if original_output is not None and compressed_output is not None:
            if original_output != compressed_output:
                kl_divergence = 0.5 
        
        kl_penalty = -gamma * kl_divergence
        
        c_s = 0.3  
        c_l = 0.8  
        
        band_penalty = 0.0
        if rho < c_s:
            band_penalty = -2.0 * (c_s - rho)
        elif rho > c_l:
            band_penalty = -2.0 * (rho - c_l)
        
        total_reward = compression_reward + semantic_reward + kl_penalty + band_penalty
        
        return total_reward, {
            "compression_reward": compression_reward,
            "semantic_reward": semantic_reward,
            "kl_penalty": kl_penalty,
            "band_penalty": band_penalty,
            "rho": rho,
            "total": total_reward
        }
    
    def ppo_update(
        self, 
        trajectories, 
        optimizer, 
        clip_ratio=0.2, 
        value_coef=0.5, 
        entropy_coef=0.01, 
        epochs=4
    ):
        """
        Update the policy using Proximal Policy Optimization (PPO)
        
        Args:
            trajectories: List of (state, action, log_prob, reward, value) tuples
            optimizer: Optimizer for updating network parameters
            clip_ratio: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            epochs: Number of optimization epochs
        """
        states, actions, old_log_probs, rewards, old_values = zip(*trajectories)
        
        rewards = torch.tensor(rewards, device=self.device)
        old_values = torch.cat(old_values)
        old_log_probs = torch.cat(old_log_probs)
        
        advantages = rewards - old_values.squeeze()
        returns = rewards
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            indices = torch.randperm(len(states))
            
            for i in indices:
                state = states[i]
                action = actions[i]
                
                policy_logits, value = self.forward(state)
                
                log_probs = F.log_softmax(policy_logits, dim=-1)
                curr_log_probs = torch.gather(
                    log_probs, -1, 
                    action.unsqueeze(-1)
                ).squeeze(-1)
                
                ratio = torch.exp(curr_log_probs - old_log_probs[i])
                clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                
                policy_loss = -torch.min(
                    ratio * advantages[i],
                    clipped_ratio * advantages[i]
                ).mean()
                
                value_loss = F.mse_loss(value.squeeze(), returns[i])
                
                probs = F.softmax(policy_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def train(
        self,
        prompts,
        model=None,  
        epochs=100,
        lr=1e-4,
        target_ratio=0.5,
        max_steps=3,
        temperature=1.0,
        batch_size=8
    ):
        """
        Train the DPC agent using PPO
        
        Args:
            prompts: List of prompt examples for training
            model: LLM to use for output evaluation (optional)
            epochs: Number of training epochs
            lr: Learning rate
            target_ratio: Target compression ratio
            max_steps: Maximum compression steps
            temperature: Policy sampling temperature
            batch_size: Batch size for training
        """
        optimizer = Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            batch_indices = np.random.choice(len(prompts), min(batch_size, len(prompts)), replace=False)
            batch_prompts = [prompts[i] for i in batch_indices]
            
            total_reward = 0.0
            all_compression_ratios = []
            
            for prompt in batch_prompts:
                trajectories = []
                
                encoded = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.max_length
                ).to(self.device)
                
                current_ids = encoded.input_ids
                original_length = current_ids.size(1)
                
                original_output = None
                if model is not None:
                    original_output = model.generate(prompt)
                
                for step in range(max_steps):
                    action, log_probs, value = self.get_action_and_value(current_ids, temperature)
                    
                    new_ids = self.apply_mask(current_ids, action)
                    
                    new_prompt = self.tokenizer.decode(new_ids[0], skip_special_tokens=True)
                    
                    compressed_output = None
                    if model is not None:
                        compressed_output = model.generate(new_prompt)
                    
                    reward, _ = self.compute_reward(
                        prompt, 
                        new_prompt, 
                        original_output, 
                        compressed_output,
                        target_ratio=target_ratio
                    )
                    
                    trajectories.append((current_ids, action, log_probs, reward, value))
                    
                    compression_ratio = new_ids.size(1) / original_length
                    all_compression_ratios.append(compression_ratio)
                    
                    if compression_ratio <= target_ratio or new_ids.size(1) == current_ids.size(1):
                        break
                    
                    current_ids = new_ids
                    total_reward += reward
                
                self.ppo_update(trajectories, optimizer)
            
            avg_reward = total_reward / len(batch_prompts)
            avg_compression = sum(all_compression_ratios) / len(all_compression_ratios)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Avg Reward: {avg_reward:.4f}, "
                      f"Avg Compression: {avg_compression:.2f}")
    
    def save(self, path):
        """Save the agent's parameters"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'policy_head_state_dict': self.policy_head.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
        }, path)
    
    def load(self, path):
        """Load the agent's parameters"""
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.policy_head.load_state_dict(checkpoint['policy_head_state_dict'])
        self.value_head.load_state_dict(checkpoint['value_head_state_dict'])


#Mock LLM for test
class MockLLM:
    """Mock language model for testing the DPC agent"""
    
    def __init__(self, sensitivity=0.5):
        """
        Initialize the mock LLM
        
        Args:
            sensitivity: How sensitive the model is to prompt changes (0-1)
                         Higher values make the model more sensitive to compression
        """
        self.sensitivity = sensitivity
    
    def generate(self, prompt):
        """Generate a response for a prompt"""
        keywords = re.findall(r'\b\w{4,}\b', prompt.lower())
        
        if "math" in keywords or "calculate" in keywords:
            return "I'll solve this step by step and find the answer."
        elif "explain" in keywords:
            return "Let me explain this concept in detail."
        elif "recipe" in keywords:
            return "Here's a delicious recipe you can try."
        else:
            return "I'm an AI assistant here to help you."
    
    def get_output_distribution(self, prompt):
        """
        Get a mock output distribution for a prompt
        This is a simplified version for testing
        """
        keywords = re.findall(r'\b\w{4,}\b', prompt.lower())
        unique_keywords = set(keywords)
        
        distribution = {
            "math": 0.1,
            "science": 0.1,
            "history": 0.1,
            "literature": 0.1,
            "technology": 0.1,
            "art": 0.1,
            "music": 0.1,
            "philosophy": 0.1,
            "politics": 0.1,
            "economics": 0.1
        }
        
        for keyword in unique_keywords:
            if keyword in distribution:
                distribution[keyword] += 0.2
        
        total = sum(distribution.values())
        for key in distribution:
            distribution[key] /= total
        
        return distribution


def test_dpc_agent():
    """Test the DPC agent"""
    dpc_agent = DynamicPromptCompressionAgent(
        encoder_model_name="distilroberta-base",
        device="cpu"
    )
    
    llm = MockLLM()
    
    test_prompts = [
        """
        Solve the following math problem step by step.
        
        Example 1:
        Question: If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?
        Let's think step by step.
        The train travels at 60 miles per hour.
        To find the distance, I'll multiply the speed by the time.
        Distance = Speed × Time
        Distance = 60 miles/hour × 2.5 hours
        Distance = 150 miles
        Answer: 150 miles
        
        Example 2:
        Question: A store is having a 25% off sale. If a shirt originally costs $40, how much does it cost during the sale?
        Let's think step by step.
        The shirt costs $40 originally.
        The discount is 25% of $40.
        25% of $40 = 0.25 × $40 = $10
        The sale price is the original price minus the discount.
        Sale price = $40 - $10 = $30
        Answer: $30
        
        Question: John has 5 apples. He gives 2 apples to Mary and buys 3 more. How many apples does John have now?
        Let's think step by step.
        """,
        
        """
        Write a creative short story about a journey through space.
        
        Example 1:
        Title: The Cosmic Explorer
        Once upon a time, there was a brave astronaut named Elena who dreamed of exploring the furthest reaches of the galaxy. She trained for years, mastering every skill needed for deep space travel. When the day finally came, Elena boarded her spacecraft, the Stellar Voyager, with a mixture of excitement and apprehension.
        As she departed Earth's atmosphere, the blue planet shrank behind her until it was just a tiny marble in the vast blackness. Elena felt both insignificant and powerful at the same time.
        Her journey took her past Mars with its rusty red surface, through the asteroid belt where she narrowly avoided collision with ancient space rocks, and finally to Jupiter with its swirling storms and many moons.
        Upon returning to Earth, Elena shared her discoveries with the world, inspiring a new generation of cosmic explorers.
        
        Write a story about:
        A lonely satellite that gains consciousness while orbiting a distant planet.
        """,
        
        """
        Answer the following question using the REACT method.
        
        Example 1:
        Question: What is the capital of France and what famous tower is located there?
        Thought 1: I need to identify the capital of France.
        Action 1: Recall geography knowledge about France.
        Observation 1: The capital of France is Paris.
        Thought 2: Now I need to identify a famous tower in Paris.
        Action 2: Recall famous landmarks in Paris.
        Observation 2: The Eiffel Tower is a famous tower located in Paris.
        Answer: The capital of France is Paris, and the famous Eiffel Tower is located there.
        
        Question: What is the largest planet in our solar system, and how many moons does it have?
        Thought 1:
        """
    ]
    
    test_ratios = [0.8, 0.6, 0.4]
    
    print("=== Testing DPC Agent Compression ===\n")
    
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt {i+1} (Original Length: {len(prompt)} chars):")
        print(f"First 100 chars: {prompt[:100]}...\n")
        
        for ratio in test_ratios:
            compressed = dpc_agent.compress_prompt(
                prompt, 
                target_ratio=ratio,
                max_steps=3,
                temperature=0.5
            )
            
            actual_ratio = len(compressed) / len(prompt)
            print(f"  Target ratio: {ratio:.1f}, Actual ratio: {actual_ratio:.2f}")
            print(f"  Compressed length: {len(compressed)} chars")
            print(f"  First 100 chars: {compressed[:min(100, len(compressed))]}...")
            
            reward, components = dpc_agent.compute_reward(prompt, compressed)
            print(f"  Reward: {reward:.4f}")
            print(f"  Components: {components}")
            print()
    
    print("\n=== Testing DPC Agent Training ===\n")
    
    training_prompts = test_prompts + [
        "Explain how photosynthesis works in plants.",
        "Describe the process of making chocolate from cocoa beans.",
        "Write a poem about the changing seasons."
    ]
    
    dpc_agent.train(
        prompts=training_prompts,
        model=llm,
        epochs=5,  
        lr=1e-4,
        target_ratio=0.6,
        max_steps=2,
        temperature=0.8,
        batch_size=3
    )
    
    print("\n=== Compression Before vs After Training ===\n")
    
    test_prompt = """
    Explain the theory of relativity and its implications for our understanding of space and time.
    
    Include details about both special relativity and general relativity.
    
    Also discuss how Einstein's equations changed our understanding of gravity.
    """
    
    compressed = dpc_agent.compress_prompt(
        test_prompt,
        target_ratio=0.6,
        max_steps=3,
        temperature=0.5
    )
    
    actual_ratio = len(compressed) / len(test_prompt)
    print(f"Original prompt ({len(test_prompt)} chars):")
    print(test_prompt)
    print(f"\nCompressed prompt ({len(compressed)} chars, ratio: {actual_ratio:.2f}):")
    print(compressed)
    
    reward, components = dpc_agent.compute_reward(test_prompt, compressed)
    print(f"\nReward: {reward:.4f}")
    print(f"Reward components: {components}")
    
    return dpc_agent


if __name__ == "__main__":
    agent = test_dpc_agent()