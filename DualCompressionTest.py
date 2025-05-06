import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse
from tqdm import tqdm

from DualCompressionFramework import (
    load_model_and_tokenizer,
    prune_model,
    PatternLibrary,
    DynamicPromptCompressionAgent,
    Configuration,
    SuccessiveHalvingOptimizer
)

class MockDataset:
    def __init__(self, task_type="gsm8k"):
        self.task_type = task_type
        self.data = self._generate_mock_data(task_type)
    
    def _generate_mock_data(self, task_type):
        if task_type == "gsm8k":
            return [
                {
                    "input": "John has 5 apples. He gives 2 apples to Mary and buys 3 more. How many apples does John have now?",
                    "output": "6",
                    "reasoning": "John starts with 5 apples. He gives 2 to Mary, so he has 5 - 2 = 3 apples. Then he buys 3 more, so he has 3 + 3 = 6 apples."
                },
                {
                    "input": "A store is having a 25% off sale. If a shirt costs $40, how much does it cost during the sale?",
                    "output": "30",
                    "reasoning": "The discount is 25% of $40, which is 0.25 × $40 = $10. The sale price is $40 - $10 = $30."
                },
                {
                    "input": "Tom can paint a room in 4 hours. Jane can paint the same room in 6 hours. How long will it take them to paint the room together?",
                    "output": "2.4",
                    "reasoning": "Tom's rate is 1/4 room per hour. Jane's rate is 1/6 room per hour. Together, their rate is 1/4 + 1/6 = (6+4)/24 = 10/24 = 5/12 room per hour. To paint the entire room, it will take 1 ÷ (5/12) = 12/5 = 2.4 hours."
                },
                {
                    "input": "A rectangular garden is 12 meters long and 8 meters wide. How many meters of fencing are needed to enclose the garden?",
                    "output": "40",
                    "reasoning": "The perimeter of a rectangle is 2 × (length + width). So the amount of fencing needed is 2 × (12 + 8) = 2 × 20 = 40 meters."
                },
                {
                    "input": "Emily has twice as many marbles as Jack. Jack has 15 marbles. How many marbles do they have in total?",
                    "output": "45",
                    "reasoning": "Jack has 15 marbles. Emily has twice as many as Jack, so she has 2 × 15 = 30 marbles. Together they have 15 + 30 = 45 marbles."
                },
                {
                    "input": "A car travels at a speed of 60 kilometers per hour. How far can it travel in 2.5 hours?",
                    "output": "150",
                    "reasoning": "Distance = speed × time. The car travels at 60 km/h for 2.5 hours, so the distance is 60 × 2.5 = 150 kilometers."
                },
                {
                    "input": "A baker uses 3/4 of a cup of sugar to make a batch of cookies. If she wants to make 5 batches, how many cups of sugar will she need?",
                    "output": "3.75",
                    "reasoning": "For each batch, she needs 3/4 cup of sugar. For 5 batches, she needs 5 × 3/4 = 15/4 = 3.75 cups of sugar."
                },
                {
                    "input": "A train leaves the station at 9:00 AM and travels at 80 km/h. Another train leaves the same station at 10:30 AM traveling at 100 km/h in the same direction. At what time will the second train catch up to the first train?",
                    "output": "3:00",
                    "reasoning": "By 10:30 AM, the first train has traveled for 1.5 hours at 80 km/h, covering 1.5 × 80 = 120 km. Let's say the second train catches up after t hours of travel. In that time, the first train has traveled for 1.5 + t hours, covering (1.5 + t) × 80 km. The second train has traveled for t hours, covering t × 100 km. When they meet, (1.5 + t) × 80 = t × 100. Simplifying: 120 + 80t = 100t. So 120 = 20t, and t = 6 hours. That means the second train catches up 6 hours after its departure at 10:30 AM, which is 4:30 PM."
                },
                {
                    "input": "If 8 workers can build a wall in 10 days, how many days would it take 5 workers to build the same wall, assuming all workers work at the same rate?",
                    "output": "16",
                    "reasoning": "The work done is proportional to the number of workers multiplied by the time. So (8 workers × 10 days) = (5 workers × x days), where x is the time it takes 5 workers. Solving for x: 80 = 5x, so x = 16 days."
                },
                {
                    "input": "A recipe calls for 2.5 cups of flour to make 2 dozen cookies. How many cups of flour are needed to make 5 dozen cookies?",
                    "output": "6.25",
                    "reasoning": "The recipe uses 2.5 cups of flour for 2 dozen cookies. For 5 dozen cookies, we need (5/2) times the amount of flour. So we need (5/2) × 2.5 = 6.25 cups of flour."
                }
            ]
        elif task_type == "fever":
            return [
                {
                    "input": "Claim: The Eiffel Tower is located in Rome.",
                    "output": "REFUTES",
                    "reasoning": "The Eiffel Tower is located in Paris, France, not Rome. Therefore, the claim is false."
                },
                {
                    "input": "Claim: Barack Obama was the 44th President of the United States.",
                    "output": "SUPPORTS",
                    "reasoning": "Barack Obama served as the 44th President of the United States from 2009 to 2017. Therefore, the claim is true."
                },
                {
                    "input": "Claim: The chemical symbol for water is H2O.",
                    "output": "SUPPORTS",
                    "reasoning": "The chemical formula for water is H2O, representing two hydrogen atoms and one oxygen atom. The claim is correct."
                },
                {
                    "input": "Claim: The Great Wall of China is visible from the Moon with the naked eye.",
                    "output": "REFUTES",
                    "reasoning": "The Great Wall of China is not visible from the Moon with the naked eye. This is a common misconception. The claim is false."
                },
                {
                    "input": "Claim: Albert Einstein was born in Germany.",
                    "output": "SUPPORTS",
                    "reasoning": "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire on 14 March 1879. The claim is true."
                },
                {
                    "input": "Claim: Tokyo is the capital of China.",
                    "output": "REFUTES",
                    "reasoning": "Tokyo is the capital of Japan, not China. Beijing is the capital of China. The claim is false."
                },
                {
                    "input": "Claim: The Earth revolves around the Sun.",
                    "output": "SUPPORTS",
                    "reasoning": "The Earth orbits or revolves around the Sun in an elliptical path. This is a basic fact of our solar system. The claim is true."
                },
                {
                    "input": "Claim: Photosynthesis is the process by which plants convert sunlight into chemical energy.",
                    "output": "SUPPORTS",
                    "reasoning": "Photosynthesis is indeed the process by which plants, algae, and certain bacteria convert sunlight, carbon dioxide, and water into glucose (chemical energy) and release oxygen. The claim is accurate."
                },
                {
                    "input": "Claim: The human heart has five chambers.",
                    "output": "REFUTES",
                    "reasoning": "The human heart has four chambers: two atria (upper) and two ventricles (lower). Not five chambers. The claim is false."
                },
                {
                    "input": "Claim: Mount Everest is the tallest mountain on Earth.",
                    "output": "SUPPORTS",
                    "reasoning": "Mount Everest, located in the Himalayas, is the tallest mountain on Earth above sea level, with a height of 8,848.86 meters (29,031.7 feet). The claim is true."
                }
            ]
        else:
            #General QA
            return [
                {
                    "input": "What is the capital of France?",
                    "output": "Paris",
                    "reasoning": "The capital of France is Paris."
                },
                {
                    "input": "Who wrote 'Romeo and Juliet'?",
                    "output": "William Shakespeare",
                    "reasoning": "William Shakespeare wrote 'Romeo and Juliet'."
                },
                {
                    "input": "What is the chemical symbol for gold?",
                    "output": "Au",
                    "reasoning": "The chemical symbol for gold is Au, which comes from the Latin word 'aurum'."
                },
                {
                    "input": "Which planet is known as the Red Planet?",
                    "output": "Mars",
                    "reasoning": "Mars is known as the Red Planet due to its reddish appearance, which is caused by iron oxide (rust) on its surface."
                },
                {
                    "input": "What is the largest ocean on Earth?",
                    "output": "Pacific Ocean",
                    "reasoning": "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 30% of the Earth's surface."
                },
                {
                    "input": "Who painted the Mona Lisa?",
                    "output": "Leonardo da Vinci",
                    "reasoning": "The Mona Lisa was painted by Italian Renaissance artist Leonardo da Vinci between 1503 and 1519."
                },
                {
                    "input": "What is the square root of 144?",
                    "output": "12",
                    "reasoning": "The square root of 144 is 12, because 12 × 12 = 144."
                },
                {
                    "input": "What is the main ingredient in guacamole?",
                    "output": "Avocado",
                    "reasoning": "The main ingredient in guacamole is avocado. Other ingredients typically include lime juice, cilantro, onions, and sometimes tomatoes."
                },
                {
                    "input": "Who was the first woman to win a Nobel Prize?",
                    "output": "Marie Curie",
                    "reasoning": "Marie Curie was the first woman to win a Nobel Prize. She won the Nobel Prize in Physics in 1903 (shared with her husband Pierre Curie and Henri Becquerel) and later the Nobel Prize in Chemistry in 1911."
                },
                {
                    "input": "What is the freezing point of water in Celsius?",
                    "output": "0",
                    "reasoning": "The freezing point of water at standard atmospheric pressure is 0 degrees Celsius (32 degrees Fahrenheit)."
                }
            ]
    
    def get_samples(self, n):
        """Get n random samples for evaluation"""
        indices = np.random.choice(len(self.data), min(n, len(self.data)), replace=False)
        return [self.data[i] for i in indices]
    
    def get_examples(self, n):
        """Get examples for few-shot prompting (different from evaluation samples)"""
        return self.get_samples(n)
    
    def evaluate_response(self, response, reference):
        """Evaluate if the response is correct"""
        if self.task_type == "gsm8k":
            import re
            if ":" in reference and re.match(r'\d+:\d+', reference):
                time_patterns = re.findall(r'\b\d+:\d+\b', response)
                return reference in time_patterns
        
            numbers = re.findall(r'\b\d+\.?\d*\b', response)
            try:
                return any(float(num) == float(reference) for num in numbers) if numbers else False
            except ValueError:
                return reference in numbers
        
        elif self.task_type == "fever":
            response = response.lower()
            if reference == "SUPPORTS":
                return any(term in response for term in ["support", "true", "correct"])
            elif reference == "REFUTES":
                return any(term in response for term in ["refute", "false", "incorrect"])
            else:  #NOT ENOUGH INFO
                return any(term in response for term in ["not enough", "insufficient"])
        
        else:
            return reference.lower() in response.lower()


class MockModel:
    def __init__(self, name, performance_profile="average"):
        self.name = name
        self.performance_profile = performance_profile
        self.quantized = False
        self.quantization_bits = None
        self.pruned = False
        self.sparsity_ratio = 0.0
    
    def quantize(self, bits):
        """Apply quantization"""
        self.quantized = True
        self.quantization_bits = bits
        return self
    
    def prune(self, sparsity_ratio):
        """Apply pruning"""
        self.pruned = True
        self.sparsity_ratio = sparsity_ratio
        return self
    
    def generate(self, prompt, task=None):
        """Generate a mock response"""
        if task is None:
            if "How many" in prompt or "calculate" in prompt.lower():
                task = "gsm8k"
            elif "Claim:" in prompt:
                task = "fever"
            else:
                task = "qa"
        
        if self.performance_profile == "high":
            base_accuracy = 0.85
        elif self.performance_profile == "average":
            base_accuracy = 0.70
        else:  #low
            base_accuracy = 0.55
        
        if self.quantized:
            if self.quantization_bits == 8:
                accuracy_factor = 0.95
            elif self.quantization_bits == 4:
                accuracy_factor = 0.85
            else: #2-bit
                accuracy_factor = 0.70
        else:
            accuracy_factor = 1.0
        
        if self.pruned:
            pruning_factor = 1.0 - (self.sparsity_ratio * 0.5)
        else:
            pruning_factor = 1.0
        
        pattern_factor = 1.0
        if "step by step" in prompt.lower():
            pattern_factor = 1.2  
        elif "Thought" in prompt and "Action" in prompt:
            pattern_factor = 1.15  

        effective_accuracy = min(0.95, base_accuracy * accuracy_factor * pruning_factor * pattern_factor)
        
        if np.random.random() < effective_accuracy:
            if task == "gsm8k":
                return "I'll solve this step by step. The answer is 6."
            elif task == "fever":
                if "Eiffel Tower" in prompt and "Rome" in prompt:
                    return "This claim is false. The Eiffel Tower is in Paris, not Rome."
                elif "Obama" in prompt and "44th President" in prompt:
                    return "This claim is true. Barack Obama was indeed the 44th President."
                else:
                    return "Based on my knowledge, this is true."
            else:
                if "capital of France" in prompt:
                    return "The capital of France is Paris."
                elif "Romeo and Juliet" in prompt:
                    return "William Shakespeare wrote 'Romeo and Juliet'."
                else:
                    return "I'm not sure about this question."
        else:
            if task == "gsm8k":
                wrong_answers = ["The answer is 5.", "The answer is 7.", "The answer is 8."]
                return np.random.choice(wrong_answers)
            elif task == "fever":
                wrong_answers = ["This is true.", "This is false.", "There's not enough information."]
                return np.random.choice(wrong_answers)
            else:
                wrong_answers = ["I'm not sure.", "I don't know.", "The answer is unknown."]
                return np.random.choice(wrong_answers)


def evaluate_configuration(config, dataset, verbose=True):
    """Evaluate a configuration on a dataset"""
    if verbose:
        print(f"\nEvaluating configuration: {config}")
    
    model = MockModel(config["model"], 
                     "high" if "70b" in config["model"] else "average")
    
    if config["quantization"] is not None:
        model.quantize(config["quantization"])
    
    if config["sparsity"] > 0:
        model.prune(config["sparsity"])
    
    pattern_library = PatternLibrary()
    pattern = pattern_library.get_pattern(config["pattern"])
    
    dpc_agent = None
    if config["compression"] < 1.0:
        dpc_agent = DynamicPromptCompressionAgent(
            encoder_model_name="distilroberta-base",  #use smaller model for testing
            device="cpu"
        )
    
    test_samples = dataset.get_samples(10)  
    correct = 0
    total_time = 0
    
    for sample in tqdm(test_samples, desc="Evaluating", disable=not verbose):
        if config["n_shots"] > 0:
            examples_text = ""
            examples = dataset.get_examples(config["n_shots"])
            
            if config["pattern"] == "cot":
                for i, ex in enumerate(examples):
                    examples_text += f"Example {i+1}:\nQuestion: {ex['input']}\n"
                    examples_text += f"Let's think step by step.\n{ex['reasoning']}\n"
                    examples_text += f"Answer: {ex['output']}\n\n"
            elif config["pattern"] == "react":
                for i, ex in enumerate(examples):
                    examples_text += f"Example {i+1}:\nTask: {ex['input']}\n"
                    steps = ex['reasoning'].split('\n')
                    for j in range(0, len(steps), 3):
                        examples_text += f"Thought {j//3 + 1}: {steps[j] if j < len(steps) else ''}\n"
                        examples_text += f"Action {j//3 + 1}: {steps[j+1] if j+1 < len(steps) else ''}\n"
                        examples_text += f"Observation {j//3 + 1}: {steps[j+2] if j+2 < len(steps) else ''}\n"
                    examples_text += f"Answer: {ex['output']}\n\n"
        else:
            examples_text = ""
        
        prompt = pattern.format(
            instruction=pattern.parameters["instruction"],
            examples=examples_text,
            input=sample["input"]
        )
        
        if config["compression"] < 1.0 and dpc_agent is not None:
            #Simplified for mock test
            prompt = "Compressed prompt would be applied here"  
                
        if "70b" in config["model"]:
            base_runtime = 0.05
        elif "granite" in config["model"]:
            base_runtime = 0.03
        else:
            base_runtime = 0.01
        
        runtime_factor = 1.0

        #quantization
        if config["quantization"] is not None:
            if config["quantization"] == 8:
                runtime_factor *= 0.9
            elif config["quantization"] == 4:
                runtime_factor *= 0.7
        
        #pruning speedup
        if config["sparsity"] > 0:
            runtime_factor *= (1.0 - config["sparsity"] * 0.3)
        
        #compression speedup
        if config["compression"] < 1.0:
            runtime_factor *= (0.5 + config["compression"] * 0.5) 
        
        final_runtime = base_runtime * runtime_factor
        
        start_time = time.time()
        
        time.sleep(final_runtime)

        response = model.generate(prompt, task=dataset.task_type)
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        if dataset.evaluate_response(response, sample["output"]):
            correct += 1
    
    #Metrics
    accuracy = correct / len(test_samples)
    avg_runtime = total_time / len(test_samples)
    
    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average runtime: {avg_runtime:.4f} seconds")
    
    return {
        "accuracy": accuracy,
        "runtime": avg_runtime,
        "config": config
    }


def run_test_grid():
    """Run a grid search of configurations"""
    dataset = MockDataset(task_type="gsm8k")
    
    models = ["llama-3.1-8b", "granite-3.1-8b", "llama-3.1-70b"]
    patterns = ["zero_shot", "cot", "react"]
    n_shots = [0, 3, 5]
    quantization = [None, 8, 4]
    sparsity = [0.0, 0.3, 0.5]
    compression = [1.0, 0.8, 0.6]
    
    #For simplie, test subset
    configs = []
    
    #Add baseline (no compression)
    for model in models:
        for pattern in patterns:
            for shots in n_shots:
                if pattern == "zero_shot" and shots > 0:
                    continue  
                
                configs.append({
                    "model": model,
                    "pattern": pattern,
                    "n_shots": shots,
                    "quantization": None,
                    "sparsity": 0.0,
                    "compression": 1.0
                })
    
    for model in models:
        #quantization
        for bits in [8, 4]:
            configs.append({
                "model": model,
                "pattern": "cot",
                "n_shots": 3,
                "quantization": bits,
                "sparsity": 0.0,
                "compression": 1.0
            })
        
        #pruning
        for ratio in [0.3, 0.5]:
            configs.append({
                "model": model,
                "pattern": "cot",
                "n_shots": 3,
                "quantization": None,
                "sparsity": ratio,
                "compression": 1.0
            })
        
        # Test prompt compression
        for ratio in [0.8, 0.6]:
            configs.append({
                "model": model,
                "pattern": "cot",
                "n_shots": 3,
                "quantization": None,
                "sparsity": 0.0,
                "compression": ratio
            })
        
        # Test combined compression
        configs.append({
            "model": model,
            "pattern": "cot",
            "n_shots": 3,
            "quantization": 8,
            "sparsity": 0.3,
            "compression": 0.8
        })
    
    results = []
    for config in configs:
        result = evaluate_configuration(config, dataset, verbose=False)
        results.append(result)
        print(f"Config: {config['model']}, {config['pattern']}, {config['n_shots']}-shot, "
              f"q={config['quantization']}, s={config['sparsity']}, c={config['compression']}, "
              f"Accuracy: {result['accuracy']:.4f}, Runtime: {result['runtime']:.4f}s")
    
    with open("grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    visualize_grid_results(results)
    
    return results


def visualize_grid_results(results):
    """Visualize grid search results"""
    plt.figure(figsize=(12, 10))
    
    accuracies = [r["accuracy"] for r in results]
    runtimes = [r["runtime"] for r in results]
    configs = [r["config"] for r in results]
    
    model_markers = {
        "llama-3.1-8b": "o",
        "granite-3.1-8b": "s", 
        "llama-3.1-70b": "^"
    }
    
    compression_colors = []
    for config in configs:
        if config["quantization"] is not None and config["sparsity"] > 0 and config["compression"] < 1.0:
            color = "red"  #Combined compression
        elif config["quantization"] is not None:
            color = "blue"  #Quantization only
        elif config["sparsity"] > 0:
            color = "green"  #Pruning only
        elif config["compression"] < 1.0:
            color = "purple"  #Prompt compression only
        else:
            color = "black"  #No compression
        
        compression_colors.append(color)
    
    ax = plt.subplot(111)
    
    for i, (acc, runtime, config, color) in enumerate(zip(accuracies, runtimes, configs, compression_colors)):
        marker = model_markers[config["model"]]
        ax.scatter(runtime, acc, marker=marker, color=color, s=100, alpha=0.7)
    
    for model, marker in model_markers.items():
        ax.scatter([], [], marker=marker, color="gray", s=100, label=model)
    
    ax.scatter([], [], marker="o", color="black", s=100, label="No compression")
    ax.scatter([], [], marker="o", color="blue", s=100, label="Quantization")
    ax.scatter([], [], marker="o", color="green", s=100, label="Pruning")
    ax.scatter([], [], marker="o", color="purple", s=100, label="Prompt compression")
    ax.scatter([], [], marker="o", color="red", s=100, label="Combined")
    
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Runtime for Different Compression Configurations")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig("compression_grid_results.png")
    plt.show()


def simulate_successive_halving():
    """Simulate the Successive Halving optimization"""
    dataset = MockDataset(task_type="gsm8k")
    
    configs = [
        #Baselines
        {"model": "llama-3.1-8b", "pattern": "zero_shot", "n_shots": 0, "quantization": None, "sparsity": 0.0, "compression": 1.0},
        {"model": "llama-3.1-8b", "pattern": "cot", "n_shots": 3, "quantization": None, "sparsity": 0.0, "compression": 1.0},
        {"model": "granite-3.1-8b", "pattern": "zero_shot", "n_shots": 0, "quantization": None, "sparsity": 0.0, "compression": 1.0},
        {"model": "granite-3.1-8b", "pattern": "cot", "n_shots": 3, "quantization": None, "sparsity": 0.0, "compression": 1.0},
        {"model": "llama-3.1-70b", "pattern": "zero_shot", "n_shots": 0, "quantization": None, "sparsity": 0.0, "compression": 1.0},
        
        #Quantization only
        {"model": "llama-3.1-8b", "pattern": "cot", "n_shots": 3, "quantization": 8, "sparsity": 0.0, "compression": 1.0},
        {"model": "llama-3.1-8b", "pattern": "cot", "n_shots": 3, "quantization": 4, "sparsity": 0.0, "compression": 1.0},
        
        #Pruning only
        {"model": "llama-3.1-8b", "pattern": "cot", "n_shots": 3, "quantization": None, "sparsity": 0.3, "compression": 1.0},
        {"model": "llama-3.1-8b", "pattern": "cot", "n_shots": 3, "quantization": None, "sparsity": 0.5, "compression": 1.0},
        
        #Prompt compression only
        {"model": "llama-3.1-8b", "pattern": "cot", "n_shots": 3, "quantization": None, "sparsity": 0.0, "compression": 0.8},
        {"model": "llama-3.1-8b", "pattern": "cot", "n_shots": 3, "quantization": None, "sparsity": 0.0, "compression": 0.6},
        
        #Combined compression
        {"model": "llama-3.1-8b", "pattern": "cot", "n_shots": 3, "quantization": 8, "sparsity": 0.3, "compression": 0.8},
        {"model": "llama-3.1-70b", "pattern": "cot", "n_shots": 3, "quantization": 8, "sparsity": 0.3, "compression": 0.8},
    ]
    
    remaining = configs
    rounds_data = []
    round_num = 0
    
    while len(remaining) > 1:
        round_num += 1
        print(f"\n=== Round {round_num} ===")
        print(f"Evaluating {len(remaining)} configurations")
        
        results = []
        for config in remaining:
            result = evaluate_configuration(config, dataset, verbose=False)
            results.append(result)
            print(f"Round {round_num}: {config['model']}, {config['pattern']}, {config['n_shots']}-shot, "
                 f"q={config['quantization']}, s={config['sparsity']}, c={config['compression']}, "
                 f"Accuracy: {result['accuracy']:.4f}, Runtime: {result['runtime']:.4f}s")
        
        rounds_data.append({
            "round": round_num,
            "configs": [r["config"] for r in results],
            "accuracies": [r["accuracy"] for r in results],
            "runtimes": [r["runtime"] for r in results]
        })
        
        results.sort(key=lambda x: x["accuracy"], reverse=True)
        num_to_keep = max(1, len(remaining) // 2)
        remaining = [r["config"] for r in results[:num_to_keep]]
        
        print(f"Keeping top {num_to_keep} configurations for next round")
    
    print("\n=== Final Result ===")
    final_config = remaining[0]
    final_result = evaluate_configuration(final_config, dataset, verbose=True)
    
    print(f"\nBest configuration found:")
    print(f"Model: {final_config['model']}")
    print(f"Pattern: {final_config['pattern']}")
    print(f"Shots: {final_config['n_shots']}")
    print(f"Quantization: {final_config['quantization']}")
    print(f"Sparsity: {final_config['sparsity']}")
    print(f"Compression: {final_config['compression']}")
    print(f"Accuracy: {final_result['accuracy']:.4f}")
    print(f"Runtime: {final_result['runtime']:.4f}s")
    
    visualize_successive_halving(rounds_data)
    
    return final_config


def visualize_successive_halving(rounds_data):
    """Visualize successive halving optimization results"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    
    for round_data in rounds_data:
        round_num = round_data["round"]
        accuracies = round_data["accuracies"]
        configs = round_data["configs"]
        
        x = [round_num] * len(accuracies)
        plt.scatter(x, accuracies, alpha=0.7, s=100)
        
        for i, (x_val, y_val, config) in enumerate(zip(x, accuracies, configs)):
            model_name = config["model"].split("-")[-1]  
            plt.annotate(model_name, (x_val, y_val), 
                         xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Successive Halving Rounds")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(range(1, len(rounds_data) + 1))
    
    plt.subplot(2, 1, 2)
    
    for round_data in rounds_data:
        round_num = round_data["round"]
        runtimes = round_data["runtimes"]
        configs = round_data["configs"]
        
        x = [round_num] * len(runtimes)
        plt.scatter(x, runtimes, alpha=0.7, s=100)
        
        for i, (x_val, y_val, config) in enumerate(zip(x, runtimes, configs)):
            compression_info = ""
            if config["quantization"] is not None:
                compression_info += f"Q{config['quantization']}"
            if config["sparsity"] > 0:
                compression_info += f"S{int(config['sparsity']*100)}"
            if config["compression"] < 1.0:
                compression_info += f"C{int(config['compression']*100)}"
            if not compression_info:
                compression_info = "Base"
            
            plt.annotate(compression_info, (x_val, y_val), 
                         xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Round")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Across Successive Halving Rounds")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(range(1, len(rounds_data) + 1))
    
    plt.tight_layout()
    plt.savefig("successive_halving_results.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual Compression Evaluation")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "optimize"],
                       help="Evaluation mode: grid search or successive halving optimization")
    args = parser.parse_args()
    
    if args.mode == "grid":
        print("Running grid search evaluation...")
        run_test_grid()
    else:
        print("Running successive halving optimization...")
        simulate_successive_halving()
    
    print("Evaluation complete.")