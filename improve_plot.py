import matplotlib.pyplot as plt
import numpy as np

rounds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]
models = ['llama-3.1-8b', 'llama-3.1-8b', 'granite-3.1-8b', 'granite-3.1-8b', 'llama-3.1-70b', 
          'llama-3.1-8b', 'llama-3.1-8b', 'llama-3.1-8b', 'llama-3.1-8b', 'llama-3.1-8b', 
          'llama-3.1-8b', 'llama-3.1-8b', 'llama-3.1-70b',
          'llama-3.1-8b', 'granite-3.1-8b', 'llama-3.1-70b', 'llama-3.1-8b', 'llama-3.1-8b', 'llama-3.1-8b',
          'llama-3.1-8b', 'granite-3.1-8b', 'llama-3.1-70b']

accuracies = [0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0,
              0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
              0.1, 0.1, 0.1]

runtimes = [0.0126, 0.0124, 0.0340, 0.0331, 0.0532, 0.0113, 0.0084, 0.0109, 0.0103, 0.0113, 0.0099, 0.0093, 0.0415,
            0.0124, 0.0331, 0.0537, 0.0109, 0.0111, 0.0098,
            0.0123, 0.0342, 0.0536]

patterns = ['zero_shot', 'cot', 'zero_shot', 'cot', 'zero_shot', 'cot', 'cot', 'cot', 'cot', 'cot', 'cot', 'cot', 'cot',
            'zero_shot', 'cot', 'zero_shot', 'cot', 'cot', 'cot',
            'zero_shot', 'cot', 'zero_shot']

quantizations = [None, None, None, None, None, 8, 4, None, None, None, None, 8, 8,
                 None, None, None, 8, None, None,
                 None, None, None]

sparsities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.3, 0.3,
              0.0, 0.0, 0.0, 0.0, 0.3, 0.0,
              0.0, 0.0, 0.0]

compressions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.8, 0.8,
                1.0, 1.0, 1.0, 1.0, 1.0, 0.6,
                1.0, 1.0, 1.0]

plt.figure(figsize=(15, 12))

model_colors = {
    'llama-3.1-8b': 'blue',
    'granite-3.1-8b': 'green',
    'llama-3.1-70b': 'red'
}

compression_markers = {
    'none': 'o',      #No compression
    'quant': 's',     #Quantization only
    'sparse': '^',    #Pruning only
    'prompt': 'P',    #Prompt compression only
    'combined': '*'   #Combined compression
}

def get_marker(quant, sparse, comp):
    if quant is None and sparse == 0.0 and comp == 1.0:
        return compression_markers['none']
    elif quant is not None and sparse == 0.0 and comp == 1.0:
        return compression_markers['quant']
    elif quant is None and sparse > 0.0 and comp == 1.0:
        return compression_markers['sparse']
    elif quant is None and sparse == 0.0 and comp < 1.0:
        return compression_markers['prompt']
    else:
        return compression_markers['combined']

jitter = np.random.normal(0, 0.05, len(rounds))
x_pos = np.array(rounds) + jitter

#Plot 1: Accuracy vs Round
plt.subplot(2, 1, 1)

for i in range(len(rounds)):
    marker = get_marker(quantizations[i], sparsities[i], compressions[i])
    color = model_colors[models[i]]
    
    if quantizations[i] is not None or sparsities[i] > 0.0 or compressions[i] < 1.0:
        comp_label = ""
        if quantizations[i] is not None:
            comp_label += f"Q{quantizations[i]}"
        if sparsities[i] > 0.0:
            comp_label += f"S{int(sparsities[i]*100)}"
        if compressions[i] < 1.0:
            comp_label += f"C{int(compressions[i]*100)}"
    else:
        comp_label = "Base"
    
    plt.scatter(x_pos[i], accuracies[i], 
                marker=marker, color=color, s=150, alpha=0.8,
                edgecolors='black', linewidths=1)
    
    model_size = models[i].split('-')[-1]
    plt.annotate(f"{model_size}\n{comp_label}", 
                 (x_pos[i], accuracies[i]),
                 xytext=(0, 10), 
                 textcoords='offset points',
                 ha='center', fontsize=9)

plt.title('Accuracy Across Successive Halving Rounds', fontsize=16)
plt.xlabel('Round', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks([1, 2, 3])
plt.ylim(-0.02, 0.15)
plt.grid(True, linestyle='--', alpha=0.7)

#Plot 2: Runtime vs Round
plt.subplot(2, 1, 2)

for i in range(len(rounds)):
    marker = get_marker(quantizations[i], sparsities[i], compressions[i])
    color = model_colors[models[i]]
    
    if quantizations[i] is not None or sparsities[i] > 0.0 or compressions[i] < 1.0:
        comp_label = ""
        if quantizations[i] is not None:
            comp_label += f"Q{quantizations[i]}"
        if sparsities[i] > 0.0:
            comp_label += f"S{int(sparsities[i]*100)}"
        if compressions[i] < 1.0:
            comp_label += f"C{int(compressions[i]*100)}"
    else:
        comp_label = "Base"
    
    plt.scatter(x_pos[i], runtimes[i], 
                marker=marker, color=color, s=150, alpha=0.8,
                edgecolors='black', linewidths=1)
    
    plt.annotate(f"{model_size}\n{comp_label}", 
                 (x_pos[i], runtimes[i]),
                 xytext=(0, 7), 
                 textcoords='offset points',
                 ha='center', fontsize=9)

plt.title('Runtime Across Successive Halving Rounds', fontsize=16)
plt.xlabel('Round', fontsize=14)
plt.ylabel('Runtime (seconds)', fontsize=14)
plt.xticks([1, 2, 3])
plt.grid(True, linestyle='--', alpha=0.7)

plt.figlegend(
    [plt.Line2D([0], [0], marker=marker, color='gray', linestyle='None', markersize=10) 
     for marker in compression_markers.values()],
    compression_markers.keys(),
    title="Compression Technique",
    loc="lower center", 
    bbox_to_anchor=(0.5, 0.02),
    ncol=len(compression_markers)
)

plt.figlegend(
    [plt.Line2D([0], [0], marker='o', color=color, linestyle='None', markersize=10) 
     for color in model_colors.values()],
    model_colors.keys(),
    title="Model",
    loc="lower center", 
    bbox_to_anchor=(0.5, 0.08),
    ncol=len(model_colors)
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('improved_optimization_results.png', dpi=300)
plt.show()