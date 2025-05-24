# Agentic AI: Bridging Efficiency and Flexibility in LLMs

**COMP 414: Optimization Final Project**  
Authors: Aaron Wu, Benjamin Mao, Lucas Li  

## 🔍 Overview

This project explores **dual compression techniques**—combining **model compression** and **prompt compression**—to optimize the efficiency of large language models (LLMs) without significantly compromising performance. We specifically focus on agentic AI systems that are autonomous, goal-setting, and capable of self-directed execution.

## 🧠 What is Agentic AI?

Agentic AI refers to autonomous agents powered by LLMs, memory, and planning, capable of:
- Perception
- Decision-making
- Tool use
- Self-directed execution

Example platforms: AutoGPT, BabyAGI

## ⚙️ Solution Architecture

Our proposed **Agentic Solution Architecture** includes:
- **Search Space Specification**: Converts unstructured prompts into combinatorial search problems.
- **Pattern Library**: Domain-specific language for structured prompt composition.
- **Successive Halving Optimizer**: Efficient optimization to avoid brute-force search.

## 🧩 Dual Compression Strategy

### 1. Model Compression
Techniques:
- **Quantization**
- **Pruning**

Goals:
- Reduce memory usage
- Speed up inference
- Enable edge deployment

### 2. Prompt Compression
Approach:
- Optimize prompts to recover or retain model performance post-compression
- Use reinforcement learning (e.g., PPO) for dynamic prompt generation

## 📊 Key Results

- Compression reduces resource cost by over 50%
- Carefully compressed prompts recover performance within 1–3% of baseline
- Pruned LLAMA-7B with a learned prompt shows comparable accuracy to full models

## 🧪 Methodology

- Used PPO to train prompt agents
- Designed reward functions based on performance degradation
- Modeled problem as an MDP
- Evaluated with dynamic compression across task domains

## 🔮 Future Work

- Develop a **unified optimization framework** for both model and prompt compression
- Validate across **multiple model families** and **task types**
- Maintain agent autonomy while minimizing compute cost

## 📚 References

Key papers:
- [Compress, Then Prompt](https://doi.org/10.48550/arXiv.2305.11186)
- [Dynamic Compressing Prompts](https://doi.org/10.48550/arXiv.2504.11004)
- [AutoPDL](https://doi.org/10.48550/arXiv.2504.04365)
- [TACO-RL](https://doi.org/10.48550/arXiv.2409.13035)

See full reference list in the presentation.

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/agentic-ai-dual-compression.git
cd agentic-ai-dual-compression
