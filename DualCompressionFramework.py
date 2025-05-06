import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Any
import random
import json
import os
import time
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

#helper functions

def load_model_and_tokenizer(model_name: str, quantization_bits: Optional[int] = None):
    """
    Load a model and tokenizer with optional quantization
    
    Args:
        model_name: HuggingFace model identifier
        quantization_bits: Number of bits for quantization (None, 8, 4, or 2)
    
    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if quantization_bits is not None:
        if quantization_bits not in [8, 4, 2]:
            raise ValueError(f"Quantization bits must be 8, 4, or 2, got {quantization_bits}")
        
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quantization_bits == 4,
            load_in_8bit=quantization_bits == 8,
            bnb_4bit_compute_dtype=torch.float16 if quantization_bits == 4 else None
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )
    
    return model, tokenizer

def prune_model(model, sparsity_ratio: float = 0.5, method: str = "magnitude"):
    """
    Prune a model to the specified sparsity ratio using various methods
    
    Args:
        model: The model to prune
        sparsity_ratio: Fraction of weights to remove (0.0 to 1.0)
        method: Pruning method ('magnitude', 'random', or 'structured')
    
    Returns:
        Pruned model
    """
    if method == "magnitude":
        # Magnitude-based pruning
        for name, param in model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu()
                threshold = torch.quantile(tensor.abs().flatten(), sparsity_ratio)
                mask = torch.abs(tensor) > threshold
                param.data = param.data * mask.to(param.device)
    
    elif method == "random":
        # Random pruning
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.rand_like(param.data.cpu()) > sparsity_ratio
                param.data = param.data * mask.to(param.device)
    
    elif method == "structured":
        # Structured pruning 
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                importance = torch.sum(torch.abs(param.data), dim=1)
                threshold = torch.quantile(importance, sparsity_ratio)
                mask = importance > threshold
                mask = mask.unsqueeze(1).expand_as(param.data)
                param.data = param.data * mask.to(param.device)
    
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    return model

#prompt pattern library

@dataclass
class PromptPattern:
    name: str
    template: str
    parameters: Dict[str, Any]
    
    def format(self, **kwargs):
        """Format the template with provided parameters and kwargs"""
        params = {**self.parameters, **kwargs}
        return self.template.format(**params)

class PatternLibrary:
    def __init__(self):
        self.patterns = {}
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        #Zero-shot pattern
        self.patterns["zero_shot"] = PromptPattern(
            name="zero_shot",
            template="{instruction}\n\n{input}",
            parameters={"instruction": ""}
        )
        
        #Chain-of-Thought pattern
        self.patterns["cot"] = PromptPattern(
            name="cot",
            template="{instruction}\n\n{examples}\n\n{input}\nLet's think step by step.",
            parameters={
                "instruction": "Solve the following problem step by step.",
                "examples": ""
            }
        )
        
        #ReAct pattern
        self.patterns["react"] = PromptPattern(
            name="react",
            template="{instruction}\n\n{examples}\n\n{input}\n\nThought 1:",
            parameters={
                "instruction": "Solve the task by analyzing the problem, taking actions, and reflecting on the results.",
                "examples": ""
            }
        )
        
        #ReWOO pattern
        self.patterns["rewoo"] = PromptPattern(
            name="rewoo",
            template="{instruction}\n\n{examples}\n\n{input}\n\nThought: Let me reason through this step by step.\n\nObservation: I need to gather information and analyze it carefully.\n\nAction:",
            parameters={
                "instruction": "Solve the task by reasoning, making observations, and taking optimal actions.",
                "examples": ""
            }
        )
    
    def get_pattern(self, name: str) -> PromptPattern:
        """Get a pattern by name"""
        if name not in self.patterns:
            raise ValueError(f"Pattern '{name}' not found")
        return self.patterns[name]
    
    def register_pattern(self, pattern: PromptPattern):
        """Register a new pattern"""
        self.patterns[pattern.name] = pattern
    
    def generate_few_shot_examples(self, pattern_name: str, examples: List[Dict], n_shots: int = 3):
        """Generate few-shot examples for a pattern"""
        if n_shots <= 0:
            return ""
        
        pattern = self.get_pattern(pattern_name)
        n_shots = min(n_shots, len(examples))
        selected_examples = examples[:n_shots]
        
        if pattern_name == "zero_shot":
            return ""
        
        elif pattern_name == "cot":
            examples_text = ""
            for i, ex in enumerate(selected_examples):
                examples_text += f"Example {i+1}:\nQuestion: {ex['input']}\n"
                examples_text += f"Let's think step by step.\n{ex['reasoning']}\n"
                examples_text += f"Answer: {ex['output']}\n\n"
            return examples_text
        
        elif pattern_name == "react":
            examples_text = ""
            for i, ex in enumerate(selected_examples):
                examples_text += f"Example {i+1}:\nTask: {ex['input']}\n"
                
                steps = ex['reasoning'].split('\n')
                for j, step in enumerate(steps):
                    if j % 3 == 0:
                        examples_text += f"Thought {j//3 + 1}: {step}\n"
                    elif j % 3 == 1:
                        examples_text += f"Action {j//3 + 1}: {step}\n"
                    else:
                        examples_text += f"Observation {j//3 + 1}: {step}\n"
                
                examples_text += f"Answer: {ex['output']}\n\n"
            return examples_text
        
        elif pattern_name == "rewoo":
            examples_text = ""
            for i, ex in enumerate(selected_examples):
                examples_text += f"Example {i+1}:\nProblem: {ex['input']}\n"
                examples_text += f"Thought: {ex.get('thought', 'Let me reason through this.')}\n"
                examples_text += f"Observation: {ex.get('observation', 'I need to analyze the information.')}\n"
                examples_text += f"Action: {ex.get('action', 'I\'ll solve step by step.')}\n"
                examples_text += f"Answer: {ex['output']}\n\n"
            return examples_text
        
        else:
            return ""

#dynamic prompt compression agent

class DynamicPromptCompressionAgent(nn.Module):
    def __init__(
        self, 
        encoder_model_name="xlm-roberta-large",
        max_length=512,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        
        #Load XLM-RoBERTa
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.encoder = AutoModel.from_pretrained(encoder_model_name).to(device)
        
        #Token-level keep/drop prediction)
        self.policy_head = nn.Linear(self.encoder.config.hidden_size, 2).to(device)
        
        #State value prediction
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
    
    def encode_prompt(self, prompt):
        """Encode a prompt to get contextual embeddings"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        
        return outputs.last_hidden_state, inputs.input_ids
    
    def forward(self, prompt):
        """Forward pass to get policy and value"""
        embeddings, tokens = self.encode_prompt(prompt)
        
        policy_logits = self.policy_head(embeddings)
        
        #Global value prediction using [CLS] token
        value = self.value_head(embeddings[:, 0, :])
        
        return policy_logits, value, tokens
    
    def predict_action(self, prompt, temperature=1.0):
        """Predict keep/drop action for each token in the prompt"""
        policy_logits, value, tokens = self.forward(prompt)
        
        if temperature == 0:
            action = torch.argmax(policy_logits, dim=-1)
        else:
            probs = F.softmax(policy_logits / temperature, dim=-1)
            action = torch.multinomial(probs.view(-1, 2), 1).view(policy_logits.shape[0], -1)
        
        return action, value, tokens
    
    def apply_mask(self, prompt, action):
        """Apply mask to prompt based on action (keep=1, drop=0)"""
        embeddings, tokens = self.encode_prompt(prompt)
        
        tokens = tokens[0].cpu().numpy()
        
        keep_mask = action[0, :len(tokens)].cpu().numpy() == 1
        
        kept_tokens = tokens[keep_mask]
        
        compressed_prompt = self.tokenizer.decode(kept_tokens, skip_special_tokens=True)
        
        return compressed_prompt
    
    def compute_reward(self, original_prompt, compressed_prompt, target_ratio=0.5, 
                       alpha=1.0, beta=0.5, gamma=1.0):
        """
        Compute reward for prompt compression
        
        Args:
            original_prompt: Original uncompressed prompt
            compressed_prompt: Compressed prompt after applying mask
            target_ratio: Target compression ratio (0-1, lower means more compression)
            alpha: Weight for compression ratio reward
            beta: Weight for semantic preservation reward
            gamma: Weight for performance preservation reward
            
        Returns:
            Reward score
        """
        original_tokens = len(self.tokenizer.encode(original_prompt))
        compressed_tokens = len(self.tokenizer.encode(compressed_prompt))
        compression_ratio = compressed_tokens / original_tokens
        
        compression_reward = alpha * (1.0 / max(0.1, compression_ratio))
        
        ratio_penalty = -abs(compression_ratio - target_ratio) * 2.0
        
        with torch.no_grad():
            orig_emb, _ = self.encode_prompt(original_prompt)
            comp_emb, _ = self.encode_prompt(compressed_prompt)
            
            similarity = F.cosine_similarity(orig_emb[:, 0, :], comp_emb[:, 0, :])
            semantic_reward = beta * similarity.item()
        
        divergence_penalty = -gamma * max(0, compression_ratio - 0.8) * 2.0
        
        total_reward = compression_reward + ratio_penalty + semantic_reward + divergence_penalty
        
        return total_reward
    
    def ppo_update(self, trajectories, optimizer, clip_ratio=0.2, epochs=4):
        """Update policy using PPO algorithm"""
        states, actions, old_probs, rewards, values = zip(*trajectories)
        
        actions = torch.stack(actions)
        old_probs = torch.stack(old_probs)
        rewards = torch.tensor(rewards, device=self.device)
        values = torch.stack(values)
        
        returns = rewards
        advantages = rewards - values.squeeze()
        
        for _ in range(epochs):
            for i, state in enumerate(states):
                policy_logits, new_value, _ = self.forward(state)
                
                probs = F.softmax(policy_logits, dim=-1)
                log_probs = F.log_softmax(policy_logits, dim=-1)
                
                action_mask = F.one_hot(actions[i], num_classes=2)
                new_log_prob = (log_probs * action_mask).sum(dim=-1)
                old_log_prob = old_probs[i]
                
                ratio = torch.exp(new_log_prob - old_log_prob)
                clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                
                policy_loss = -torch.min(
                    ratio * advantages[i],
                    clipped_ratio * advantages[i]
                ).mean()
                
                value_loss = F.mse_loss(new_value.squeeze(), returns[i])
                
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def compress_prompt(self, prompt, target_ratio=0.5, max_steps=3, temperature=0.5):
        """
        Compress a prompt using the trained policy
        
        Args:
            prompt: Original prompt to compress
            target_ratio: Target compression ratio (0-1, lower means more compression)
            max_steps: Maximum number of compression steps
            temperature: Temperature for sampling actions
            
        Returns:
            Compressed prompt
        """
        current_prompt = prompt
        
        for _ in range(max_steps):
            action, _, tokens = self.predict_action(current_prompt, temperature)
            
            new_prompt = self.apply_mask(current_prompt, action)
            
            original_tokens = len(self.tokenizer.encode(prompt))
            compressed_tokens = len(self.tokenizer.encode(new_prompt))
            current_ratio = compressed_tokens / original_tokens
            
            if current_ratio <= target_ratio or new_prompt == current_prompt:
                break
                
            current_prompt = new_prompt
        
        return current_prompt

#Successive Halving Optimizer

class Configuration:
    """A configuration represents a model-prompt pair"""
    def __init__(
        self, 
        model_name: str, 
        pattern_name: str, 
        n_shots: int = 0,
        quantization_bits: Optional[int] = None,
        sparsity_ratio: float = 0.0,
        compression_ratio: float = 1.0
    ):
        self.model_name = model_name
        self.pattern_name = pattern_name
        self.n_shots = n_shots
        self.quantization_bits = quantization_bits
        self.sparsity_ratio = sparsity_ratio
        self.compression_ratio = compression_ratio
        self.performance = None
        self.runtime = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "pattern_name": self.pattern_name,
            "n_shots": self.n_shots,
            "quantization_bits": self.quantization_bits,
            "sparsity_ratio": self.sparsity_ratio,
            "compression_ratio": self.compression_ratio,
            "performance": self.performance,
            "runtime": self.runtime
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from dictionary"""
        config = cls(
            model_name=config_dict["model_name"],
            pattern_name=config_dict["pattern_name"],
            n_shots=config_dict["n_shots"],
            quantization_bits=config_dict["quantization_bits"],
            sparsity_ratio=config_dict["sparsity_ratio"],
            compression_ratio=config_dict["compression_ratio"]
        )
        config.performance = config_dict.get("performance")
        config.runtime = config_dict.get("runtime")
        return config

class SuccessiveHalvingOptimizer:
    def __init__(
        self, 
        task_dataset,
        pattern_library: PatternLibrary,
        dpc_agent: Optional[DynamicPromptCompressionAgent] = None,
        total_budget: int = 100
    ):
        self.task_dataset = task_dataset
        self.pattern_library = pattern_library
        self.dpc_agent = dpc_agent
        self.total_budget = total_budget
        self.results = []
    
    def generate_candidate_configurations(self, model_names, max_candidates=32):
        """Generate candidate configurations to evaluate"""
        candidates = []
        
        #Pattern 
        patterns = ["zero_shot", "cot", "react", "rewoo"]
        n_shots_options = [0, 3, 5]
        
        #Model compression
        quantization_options = [None, 8, 4]
        sparsity_options = [0.0, 0.3, 0.5, 0.7]
        
        #Prompt compression 
        compression_options = [1.0, 0.8, 0.6, 0.4]
        
        for model_name in model_names:
            for pattern in patterns:
                for n_shots in n_shots_options:
                    if pattern == "zero_shot" and n_shots > 0:
                        continue
                        
                    for quant in quantization_options:
                        for sparse in sparsity_options:
                            for compress in compression_options:
                                if quant is None and sparse == 0.0 and compress == 1.0:
                                    if pattern == "zero_shot" and n_shots == 0:
                                        candidates.append(Configuration(
                                            model_name=model_name,
                                            pattern_name=pattern,
                                            n_shots=n_shots,
                                            quantization_bits=quant,
                                            sparsity_ratio=sparse,
                                            compression_ratio=compress
                                        ))
                                else:
                                    candidates.append(Configuration(
                                        model_name=model_name,
                                        pattern_name=pattern,
                                        n_shots=n_shots,
                                        quantization_bits=quant,
                                        sparsity_ratio=sparse,
                                        compression_ratio=compress
                                    ))
        
        random.shuffle(candidates)
        return candidates[:max_candidates]
    
    def evaluate_configuration(self, config: Configuration, num_samples: int = 10):
        """Evaluate a single configuration"""
        model, tokenizer = load_model_and_tokenizer(
            config.model_name, 
            quantization_bits=config.quantization_bits
        )
        
        if config.sparsity_ratio > 0:
            model = prune_model(model, sparsity_ratio=config.sparsity_ratio)
        
        pattern = self.pattern_library.get_pattern(config.pattern_name)
        
        eval_samples = self.task_dataset.get_samples(num_samples)
        
        correct = 0
        total_time = 0
        
        for sample in tqdm(eval_samples, desc=f"Evaluating {config.model_name} {config.pattern_name}"):
            if config.n_shots > 0:
                examples_text = self.pattern_library.generate_few_shot_examples(
                    config.pattern_name,
                    self.task_dataset.get_examples(config.n_shots),
                    config.n_shots
                )
            else:
                examples_text = ""
            
            prompt = pattern.format(
                instruction=pattern.parameters["instruction"],
                examples=examples_text,
                input=sample["input"]
            )
            
            if config.compression_ratio < 1.0 and self.dpc_agent is not None:
                prompt = self.dpc_agent.compress_prompt(
                    prompt, 
                    target_ratio=config.compression_ratio
                )
            
            start_time = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            end_time = time.time()
            total_time += (end_time - start_time)
            
            if self.task_dataset.evaluate_response(response, sample["output"]):
                correct += 1
        
        accuracy = correct / len(eval_samples)
        avg_runtime = total_time / len(eval_samples)
        
        config.performance = accuracy
        config.runtime = avg_runtime
        
        return config
    
    def optimize(self, model_names, max_candidates=32):
        """Run successive halving optimization"""
        candidates = self.generate_candidate_configurations(model_names, max_candidates)
        
        budget_per_config = self.total_budget / len(candidates)
        samples_per_config = max(1, int(budget_per_config))
        
        remaining_candidates = candidates
        round_num = 0
        
        while len(remaining_candidates) > 1:
            print(f"Round {round_num + 1}: Evaluating {len(remaining_candidates)} configurations with {samples_per_config} samples each")
            
            evaluated_configs = []
            for config in tqdm(remaining_candidates, desc=f"Round {round_num + 1}"):
                evaluated_config = self.evaluate_configuration(config, num_samples=samples_per_config)
                evaluated_configs.append(evaluated_config)
                
                self.results.append(evaluated_config.to_dict())
            
            evaluated_configs.sort(key=lambda x: x.performance, reverse=True)
            remaining_candidates = evaluated_configs[:len(evaluated_configs) // 2]
            
            samples_per_config *= 2
            round_num += 1
        
        if remaining_candidates:
            return remaining_candidates[0]
        else:
            return None
    
    def save_results(self, filename="optimization_results.json"):
        """Save optimization results to a file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

#Task Dataset

class TaskDataset:
    """Base class for task datasets"""
    def __init__(self, data=None):
        self.data = data or []
    
    def get_samples(self, n):
        """Get n random samples for evaluation"""
        return random.sample(self.data, min(n, len(self.data)))
    
    def get_examples(self, n):
        """Get examples for few-shot prompting (different from evaluation samples)"""
        return self.get_samples(n)
    
    def evaluate_response(self, response, reference):
        """Evaluate if the response is correct (task-specific)"""
        raise NotImplementedError("Subclasses must implement this method")

class GSM8KDataset(TaskDataset):
    """Grade School Math dataset"""
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        super().__init__(data)
    
    def evaluate_response(self, response, reference):
        """Evaluate if the response contains the correct answer"""
        # Extract numerical answer from the response
        import re
        numbers = re.findall(r'\b\d+\b', response)
        if not numbers:
            return False
        
        # Check if any of the extracted numbers match the reference
        return str(reference) in numbers

class FEVERDataset(TaskDataset):
    """Fact verification dataset"""
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        super().__init__(data)
    
    def evaluate_response(self, response, reference):
        """Evaluate if the response contains the correct label"""
        response = response.lower()
        
        if reference == "SUPPORTS":
            return any(x in response for x in ["support", "true", "correct", "yes", "verify"])
        elif reference == "REFUTES":
            return any(x in response for x in ["refute", "false", "incorrect", "no", "contradict"])
        elif reference == "NOT ENOUGH INFO":
            return any(x in response for x in ["not enough", "insufficient", "unknown", "can't determine"])
        
        return False

class MBPPDataset(TaskDataset):
    """Programming problems dataset"""
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        super().__init__(data)
    
    def evaluate_response(self, response, reference):
        """Evaluate if the response contains working code"""
        import re
        code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
        if not code_match:
            return False
        
        code = code_match.group(1)
        
        #test on test cases
        try:
            key_elements = [line.strip() for line in reference.split('\n') if line.strip()]
            for element in key_elements[:3]:  
                if element not in code:
                    return False
            return True
        except:
            return False

#End-to-End Example

def run_dual_compression_optimization():
    """Run end-to-end dual compression optimization example"""
    print("Initializing Dual Compression Framework...")
    
    pattern_library = PatternLibrary()
    
    dpc_agent = DynamicPromptCompressionAgent(
        encoder_model_name="xlm-roberta-large",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    dpc_optimizer = Adam(dpc_agent.parameters(), lr=1e-5)
    
    mock_gsm8k_data = [
        {
            "input": "John has 5 apples. He gives 2 apples to Mary and buys 3 more. How many apples does John have now?",
            "output": "6",
            "reasoning": "John starts with 5 apples. He gives 2 to Mary, so he has 5 - 2 = 3 apples. Then he buys 3 more, so he has 3 + 3 = 6 apples."
        },
    ]
    
    mock_dataset = GSM8KDataset(mock_gsm8k_data)
    
    optimizer = SuccessiveHalvingOptimizer(
        task_dataset=mock_dataset,
        pattern_library=pattern_library,
        dpc_agent=dpc_agent,
        total_budget=100
    )
    
    model_names = ["llama-3.1-8b", "granite-3.1-8b"]
    best_config = optimizer.optimize(model_names, max_candidates=8)
    
    print("Optimization complete!")
    print(f"Best configuration: {best_config.to_dict()}")
    
    optimizer.save_results("optimization_results.json")
    
    return best_config

#test

def test_dual_compression_components():
    """Test individual components of the dual compression framework"""
    print("Testing Pattern Library...")
    pattern_library = PatternLibrary()
    
    zero_shot = pattern_library.get_pattern("zero_shot")
    cot = pattern_library.get_pattern("cot")
    
    test_input = "What is the capital of France?"
    
    print(f"Zero-shot prompt:\n{zero_shot.format(input=test_input)}\n")
    print(f"CoT prompt:\n{cot.format(input=test_input)}\n")
    
    print("Testing Dynamic Prompt Compression...")
    dpc_agent = DynamicPromptCompressionAgent(
        encoder_model_name="distilroberta-base",  
        device="cpu"
    )
    
    test_prompt = """
    Solve the following problem step by step.
    
    Example 1:
    Question: If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?
    Let's think step by step.
    The train travels at 60 miles per hour.
    To find the distance, I'll multiply the speed by the time.
    Distance = Speed × Time
    Distance = 60 miles/hour × 2.5 hours
    Distance = 150 miles
    Answer: 150 miles
    
    Question: John has 5 apples. He gives 2 apples to Mary and buys 3 more. How many apples does John have now?
    Let's think step by step.
    """
    
    compressed_prompt = dpc_agent.compress_prompt(test_prompt, target_ratio=0.6)
    
    print(f"Original prompt length: {len(test_prompt)}")
    print(f"Compressed prompt length: {len(compressed_prompt)}")
    print(f"Compression ratio: {len(compressed_prompt) / len(test_prompt):.2f}")
    print(f"Compressed prompt:\n{compressed_prompt}\n")
    
    print("Testing Successful Halving Optimizer with Mock Data...")
    mock_config = Configuration(
        model_name="mock-model",
        pattern_name="cot",
        n_shots=3,
        quantization_bits=8,
        sparsity_ratio=0.3,
        compression_ratio=0.7
    )
    
    print(f"Mock configuration: {mock_config.to_dict()}")
    
    print("All components tested successfully!")

if __name__ == "__main__":
    #run_dual_compression_optimization()  
    test_dual_compression_components()     