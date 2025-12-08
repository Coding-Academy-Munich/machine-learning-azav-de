import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Functions from slides_010_what_are_llms.py

def plot_llm_timeline():
    """Plot timeline of famous LLMs"""
    llms = [
        ('GPT-2', 2019, 1.5, 'OpenAI'),
        ('GPT-3', 2020, 175, 'OpenAI'),
        ('PaLM', 2022, 540, 'Google'),
        ('ChatGPT\n(GPT-3.5)', 2022, 175, 'OpenAI'),
        ('GPT-4', 2023, 1000, 'OpenAI'),  # Estimated
        ('Claude 3', 2024, 1000, 'Anthropic'),  # Estimated
        ('Gemini', 2024, 1000, 'Google')  # Estimated
    ]

    names = [name for name, _, _, _ in llms]
    sizes = [size for _, _, size, _ in llms]
    companies = [company for _, _, _, company in llms]

    # Color by company
    company_colors = {'OpenAI': 'lightblue', 'Google': 'lightcoral', 'Anthropic': 'lightgreen'}
    colors = [company_colors[c] for c in companies]

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(names, sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add size labels
    for bar, size in zip(bars, sizes):
        width = bar.get_width()
        label = f'{size}B' if size >= 1000 else f'{size:.1f}B'
        ax.text(width + 20, bar.get_y() + bar.get_height()/2,
               label, ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Parameters (Billions)', fontsize=12)
    ax.set_title('Famous Large Language Models', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=company)
                      for company, color in company_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_llm_capabilities():
    """Plot LLM capabilities"""
    capabilities = [
        'Text\nGeneration',
        'Translation',
        'Summarization',
        'Question\nAnswering',
        'Code\nGeneration',
        'Conversation'
    ]

    # Rough capability scores (illustrative)
    scores = [95, 90, 85, 88, 80, 92]

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(capabilities, scores, alpha=0.7, color='steelblue', edgecolor='black', linewidth=2)

    # Add score labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
               f'{score}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Capability Level (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title('LLM Capabilities (Illustrative)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_training_data_sources():
    """Plot sources of training data"""
    sources = ['Books', 'Websites', 'Wikipedia', 'Academic\nPapers', 'Synthetic\nData', 'Code\nRepositories']
    percentages = [1, 65, 2, 10, 4, 18]
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']

    fig, ax = plt.subplots(figsize=(7, 7))

    wedges, texts, autotexts = ax.pie(percentages, labels=sources, colors=colors,
                                       autopct='%1.0f%%', startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax.set_title('LLM Training Data Sources (Dolma 3 corpus)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_training_stages():
    """Visualize pre-training vs fine-tuning"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pre-training
    ax1.text(0.5, 0.8, 'Pre-Training', ha='center', fontsize=16, fontweight='bold',
            transform=ax1.transAxes)
    ax1.text(0.5, 0.6, '• Huge datasets\n• Months of training\n• Millions of $$$\n• Learns language',
            ha='center', va='top', fontsize=12, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))
    ax1.text(0.5, 0.25, '📚  →  🤖  →  🧠', ha='center', fontsize=40, transform=ax1.transAxes)
    ax1.axis('off')

    # Fine-tuning
    ax2.text(0.5, 0.8, 'Fine-Tuning', ha='center', fontsize=16, fontweight='bold',
            transform=ax2.transAxes)
    ax2.text(0.5, 0.6, '• Specific datasets\n• Days of training\n• Thousands of $\n• Learns tasks',
            ha='center', va='top', fontsize=12, transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, pad=1))
    ax2.text(0.5, 0.25, '🧠  →  ⚙️  →  💬', ha='center', fontsize=40, transform=ax2.transAxes)
    ax2.axis('off')

    plt.suptitle('Two Stages of LLM Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_emergent_abilities():
    """Plot emergent abilities with model size"""
    model_sizes = [1, 10, 50, 100, 500, 1000]  # Billions of parameters
    basic_tasks = [40, 70, 85, 90, 92, 93]
    complex_reasoning = [0, 0, 10, 40, 75, 85]
    math_problems = [0, 0, 0, 5, 50, 80]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(model_sizes, basic_tasks, 'o-', linewidth=2, markersize=8,
           label='Basic Tasks', color='lightblue')
    ax.plot(model_sizes, complex_reasoning, 's-', linewidth=2, markersize=8,
           label='Complex Reasoning', color='lightcoral')
    ax.plot(model_sizes, math_problems, '^-', linewidth=2, markersize=8,
           label='Math Problems', color='lightgreen')

    ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=12)
    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title('Emergent Abilities with Model Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
