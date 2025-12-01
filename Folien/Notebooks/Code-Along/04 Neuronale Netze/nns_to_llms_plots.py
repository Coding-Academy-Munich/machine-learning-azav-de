"""
Plotting functions for From Neural Networks to Large Language Models slides.

All functions encapsulate data generation and plotting logic to minimize
code shown in slides. Most functions can be called without parameters.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_tokenization_example():
    """
    Visualize how text is split into tokens.
    Shows a simple tokenization example with GPT-4o token IDs.
    """
    text = "The cat sat on the mat"
    tokens = text.split()
    token_ids = [976, 9059, 10139, 402, 290, 2450]  # GPT-4o token IDs

    fig, ax = plt.subplots(figsize=(12, 4))

    # Draw boxes for each token
    box_width = 1.2
    spacing = 0.2
    colors = plt.cm.Set3(np.linspace(0, 1, len(tokens)))

    for i, (token, color, token_id) in enumerate(zip(tokens, colors, token_ids)):
        x = i * (box_width + spacing)
        rect = plt.Rectangle((x, 0.3), box_width, 0.4,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_width/2, 0.5, token, ha='center', va='center',
               fontsize=14, fontweight='bold')
        ax.text(x + box_width/2, 0.1, f'ID: {token_id}', ha='center', va='center',
               fontsize=10, color='gray')

    # Draw original text above
    total_width = len(tokens) * (box_width + spacing) - spacing
    ax.text(total_width/2, 0.9, f'"{text}"', ha='center', va='center',
           fontsize=16, style='italic')
    ax.annotate('', xy=(total_width/2, 0.75), xytext=(total_width/2, 0.85),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.set_xlim(-0.3, total_width + 0.3)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    ax.set_title('Tokenisierung / Tokenization: Text → Tokens', fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()


def plot_word_embeddings():
    """
    Visualize simplified word embeddings in 2D space.
    Shows how similar words are close together.
    """
    word_embeddings = {
        'king': np.array([0.8, 0.9]),
        'queen': np.array([0.75, 0.85]),
        'man': np.array([0.7, 0.3]),
        'woman': np.array([0.65, 0.25]),
        'dog': np.array([-0.4, 0.7]),
        'cat': np.array([-0.35, 0.65]),
        'table': np.array([-0.8, -0.6]),
        'chair': np.array([-0.75, -0.65])
    }

    plt.figure(figsize=(10, 8))
    for word, vec in word_embeddings.items():
        plt.scatter(vec[0], vec[1], s=200, alpha=0.6)
        plt.annotate(word, vec, fontsize=14, ha='center')

    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('Word Embeddings (vereinfacht / simplified)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_attention_weights():
    """
    Visualize attention weights for a simple sentence.
    Shows which words are important when processing 'chased'.
    """
    sentence = "The dog chased the cat".split()
    attention_weights = np.array([0.1, 0.4, 1.0, 0.1, 0.35])

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(sentence)), attention_weights, alpha=0.7)
    plt.xticks(range(len(sentence)), sentence, fontsize=14)
    plt.ylabel('Attention Weight / Attention-Gewicht', fontsize=12)
    plt.title('Attention für "chased" / Attention for "chased"', fontsize=14)
    plt.ylim([0, 1.2])

    # Color important words: dog and cat in red, chased in yellow
    bars[1].set_color('red')      # dog
    bars[1].set_alpha(1.0)
    bars[2].set_color('gold')     # chased
    bars[2].set_alpha(1.0)
    bars[4].set_color('red')      # cat
    bars[4].set_alpha(1.0)

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_parameter_comparison():
    """
    Compare parameter counts across different model sizes.
    Shows the scale from simple networks to LLMs.
    """
    models = ['Unsere Netze\nOur Networks', 'GPT-2', 'GPT-3', 'GPT-4\n(geschätzt)']
    params_display = [10, 1500, 175000, 1000000]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, params_display, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.yscale('log')
    plt.ylabel('Parameter (Millionen) / Parameters (millions)', fontsize=12)
    plt.title('Größenvergleich: Parameter / Size Comparison: Parameters', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    for bar, param in zip(bars, params_display):
        height = bar.get_height()
        if param >= 1000:
            label = f'{param/1000:.0f}B'
        else:
            label = f'{param:.0f}M'
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()
