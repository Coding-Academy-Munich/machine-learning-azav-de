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
    sources = ['Books', 'Wikipedia', 'Websites', 'Academic\nPapers', 'Code\nRepositories', 'Other']
    percentages = [15, 10, 50, 10, 10, 5]
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']

    fig, ax = plt.subplots(figsize=(10, 10))

    wedges, texts, autotexts = ax.pie(percentages, labels=sources, colors=colors,
                                       autopct='%1.0f%%', startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax.set_title('LLM Training Data Sources (Illustrative)', fontsize=14, fontweight='bold', pad=20)
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


# Functions from slides_020_how_llms_work.py

def plot_next_word_probabilities(prompt, probabilities):
    """Visualize probabilities for next word"""
    words = list(probabilities.keys())
    probs = list(probabilities.values())

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(words, probs, alpha=0.7, color='steelblue', edgecolor='black', linewidth=2)

    # Highlight most likely word
    bars[0].set_color('lightcoral')

    # Add probability labels
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{prob:.0%}', ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title(f'Next Word Probabilities after: "{prompt}"',
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.4)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_temperature_effect():
    """Visualize effect of temperature on probabilities"""
    words = ['mat', 'floor', 'chair', 'table', 'roof']
    original_probs = np.array([0.35, 0.25, 0.15, 0.10, 0.08])

    # Apply temperature
    def apply_temperature(probs, temp):
        logits = np.log(probs + 1e-10)
        logits = logits / temp
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    low_temp = apply_temperature(original_probs, 0.5)
    high_temp = apply_temperature(original_probs, 2.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Low temperature
    axes[0].barh(words, low_temp, alpha=0.7, color='lightblue', edgecolor='black', linewidth=2)
    axes[0].set_xlabel('Probability', fontsize=11)
    axes[0].set_title('Low Temperature (0.5)\nMore Predictable', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='x')

    # Original
    axes[1].barh(words, original_probs, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=2)
    axes[1].set_xlabel('Probability', fontsize=11)
    axes[1].set_title('Original (1.0)\nBalanced', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='x')

    # High temperature
    axes[2].barh(words, high_temp, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=2)
    axes[2].set_xlabel('Probability', fontsize=11)
    axes[2].set_title('High Temperature (2.0)\nMore Creative', fontsize=12, fontweight='bold')
    axes[2].set_xlim(0, 1)
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()


def plot_transformer_block():
    """Visualize a simplified transformer block"""
    fig, ax = plt.subplots(figsize=(10, 12))

    # Components
    components = [
        ('Input Embeddings', 0.9, 'lightblue'),
        ('Multi-Head\nAttention', 0.7, 'lightcoral'),
        ('Add & Normalize', 0.55, 'lightyellow'),
        ('Feed-Forward\nNetwork', 0.4, 'lightgreen'),
        ('Add & Normalize', 0.25, 'lightyellow'),
        ('Output', 0.1, 'lightblue')
    ]

    for name, y, color in components:
        rect = plt.Rectangle((0.2, y - 0.06), 0.6, 0.12,
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y, name, ha='center', va='center',
               fontsize=12, fontweight='bold')

        # Arrow to next component
        if y > 0.15:
            ax.arrow(0.5, y - 0.07, 0, -0.06,
                    head_width=0.05, head_length=0.02,
                    fc='black', ec='black', linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Simplified Transformer Block', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_positional_encoding_concept():
    """Visualize concept of positional encoding"""
    words = ['The', 'cat', 'sat', 'on', 'mat']
    positions = list(range(len(words)))

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Without positional encoding
    for i, word in enumerate(words):
        ax1.text(i, 0, word, ha='center', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
    ax1.set_xlim(-0.5, len(words) - 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_title('Without Positional Encoding\n(Order information lost)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # With positional encoding
    for i, (word, pos) in enumerate(zip(words, positions)):
        # Word
        ax2.text(i, 0.15, word, ha='center', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
        # Position
        ax2.text(i, -0.15, f'Pos {pos}', ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=1))
        # Connection
        ax2.plot([i, i], [0.05, -0.05], 'k--', linewidth=1.5, alpha=0.5)

    ax2.set_xlim(-0.5, len(words) - 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_title('With Positional Encoding\n(Order preserved)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def plot_context_windows():
    """Compare context window sizes"""
    models = ['GPT-3.5', 'GPT-4\n(8K)', 'GPT-4\n(32K)', 'Claude 3']
    tokens = [4_000, 8_000, 32_000, 200_000]
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(models, tokens, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add labels
    for bar, token_count in zip(bars, tokens):
        width = bar.get_width()
        if token_count >= 1000:
            label = f'{token_count/1000:.0f}K'
        else:
            label = f'{token_count}'
        ax.text(width + 5000, bar.get_y() + bar.get_height()/2,
               label, ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Context Window (Tokens)', fontsize=12)
    ax.set_title('Context Window Sizes', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_tokenization_example():
    """Visualize tokenization example"""
    text = "unbelievable performance!"
    tokens = ["un", "believ", "able", "performance", "!"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Original text
    ax.text(0.5, 0.8, f'Original Text: "{text}"', ha='center', fontsize=13,
           transform=ax.transAxes, fontweight='bold')

    # Tokens
    ax.text(0.5, 0.6, 'Tokens:', ha='center', fontsize=12,
           transform=ax.transAxes, fontweight='bold')

    x_positions = np.linspace(0.15, 0.85, len(tokens))
    for i, (token, x) in enumerate(zip(tokens, x_positions)):
        ax.text(x, 0.4, token, ha='center', va='center', fontsize=11,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
        # Token ID
        ax.text(x, 0.25, f'ID: {1000+i}', ha='center', va='center', fontsize=9,
               transform=ax.transAxes, style='italic', color='gray')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Tokenization Example', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


# Functions from slides_030_prompt_engineering.py

def plot_prompt_quality_spectrum():
    """Visualize spectrum of prompt quality"""
    fig, ax = plt.subplots(figsize=(14, 8))

    prompts = [
        ('Vague\n"Tell me about AI"', 0.2, 'red'),
        ('Basic\n"Explain AI"', 0.4, 'orange'),
        ('Specific\n"Explain AI for beginners"', 0.6, 'yellow'),
        ('Detailed\n"Explain AI for beginners\nwith examples"', 0.8, 'lightgreen'),
        ('Optimized\n"You are an expert teacher.\nExplain AI to beginners\nwith 2 real-world examples"', 1.0, 'green')
    ]

    for text, x, color in prompts:
        # Circle
        circle = plt.Circle((x, 0.5), 0.08, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)

        # Text
        ax.text(x, 0.25, text, ha='center', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))

        # Arrow
        if x < 1.0:
            ax.arrow(x + 0.09, 0.5, 0.09, 0,
                    head_width=0.04, head_length=0.02,
                    fc='black', ec='black', linewidth=1.5)

    # Axis
    ax.plot([0.1, 1.1], [0.5, 0.5], 'k-', linewidth=2, alpha=0.3)
    ax.text(0.1, 0.7, 'Poor Quality', ha='left', fontsize=12, fontweight='bold', color='red')
    ax.text(1.1, 0.7, 'High Quality', ha='right', fontsize=12, fontweight='bold', color='green')

    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Prompt Quality Spectrum', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_shot_learning_comparison():
    """Compare zero-shot and few-shot learning"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Zero-shot
    ax1.text(0.5, 0.8, 'Zero-Shot', ha='center', fontsize=16, fontweight='bold',
            transform=ax1.transAxes)
    ax1.text(0.5, 0.6, 'No examples provided', ha='center', fontsize=12,
            transform=ax1.transAxes, style='italic')
    ax1.text(0.5, 0.4, '❓ → 🤖 → ❓', ha='center', fontsize=40,
            transform=ax1.transAxes)
    ax1.text(0.5, 0.2, 'May work, but less reliable', ha='center', fontsize=11,
            transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))
    ax1.axis('off')

    # Few-shot
    ax2.text(0.5, 0.8, 'Few-Shot', ha='center', fontsize=16, fontweight='bold',
            transform=ax2.transAxes)
    ax2.text(0.5, 0.6, '2-5 examples provided', ha='center', fontsize=12,
            transform=ax2.transAxes, style='italic')
    ax2.text(0.5, 0.4, '✓ ✓ ✓ → 🤖 → ✓', ha='center', fontsize=40,
            transform=ax2.transAxes)
    ax2.text(0.5, 0.2, 'More accurate & consistent', ha='center', fontsize=11,
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    ax2.axis('off')

    plt.suptitle('Learning Strategies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_chain_of_thought():
    """Visualize chain-of-thought prompting"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Steps
    steps = [
        ('Question', 0.9, 'lightblue'),
        ('Step 1:\nBreak down', 0.7, 'lightcoral'),
        ('Step 2:\nCalculate parts', 0.5, 'lightgreen'),
        ('Step 3:\nCombine', 0.3, 'lightyellow'),
        ('Final Answer', 0.1, 'lightblue')
    ]

    for text, y, color in steps:
        rect = plt.Rectangle((0.25, y - 0.06), 0.5, 0.12,
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y, text, ha='center', va='center',
               fontsize=12, fontweight='bold')

        # Arrow
        if y > 0.15:
            ax.arrow(0.5, y - 0.07, 0, -0.06,
                    head_width=0.05, head_length=0.02,
                    fc='black', ec='black', linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Chain-of-Thought: Think Step by Step',
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_role_examples():
    """Visualize different role assignments"""
    roles = [
        ('Teacher\n👨‍🏫', 'Simple\nExplanations'),
        ('Expert\n👨‍🔬', 'Technical\nDetails'),
        ('Journalist\n📰', 'Engaging\nStories'),
        ('Comedian\n🎭', 'Humorous\nTone')
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    x_positions = np.linspace(0.15, 0.85, len(roles))

    for i, (role, style) in enumerate(roles):
        x = x_positions[i]

        # Role
        ax.text(x, 0.65, role, ha='center', va='center', fontsize=16,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue',
                        edgecolor='black', linewidth=2, pad=0.8))

        # Arrow
        ax.arrow(x, 0.57, 0, -0.07, transform=ax.transAxes,
                head_width=0.03, head_length=0.03,
                fc='black', ec='black', linewidth=1.5)

        # Style
        ax.text(x, 0.4, style, ha='center', va='center', fontsize=11,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightyellow',
                        edgecolor='black', linewidth=1, pad=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Role Assignment Changes Response Style',
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_prompt_template():
    """Visualize prompt template structure"""
    components = [
        'Role',
        'Task',
        'Context',
        'Format',
        'Examples',
        'Constraints'
    ]

    colors = ['lightblue', 'lightcoral', 'lightgreen',
             'lightyellow', 'lightpink', 'lavender']

    fig, ax = plt.subplots(figsize=(10, 10))

    y_start = 0.9
    y_step = 0.13

    for i, (component, color) in enumerate(zip(components, colors)):
        y = y_start - i * y_step

        rect = plt.Rectangle((0.15, y - 0.05), 0.7, 0.09,
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y, component, ha='center', va='center',
               fontsize=13, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Prompt Template Structure', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


# Functions from slides_040_llm_applications.py

def plot_application_areas():
    """Plot main LLM application areas"""
    applications = [
        'Content\nCreation',
        'Customer\nSupport',
        'Code\nAssistance',
        'Translation',
        'Data\nAnalysis',
        'Education',
        'Research',
        'Summarization'
    ]

    # Adoption rates (illustrative)
    adoption = [85, 75, 80, 90, 65, 70, 60, 80]

    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.bar(applications, adoption, alpha=0.7, color='steelblue',
                 edgecolor='black', linewidth=2)

    # Highlight most popular
    max_idx = np.argmax(adoption)
    bars[max_idx].set_color('lightcoral')

    # Add labels
    for bar, rate in zip(bars, adoption):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
               f'{rate}%', ha='center', va='bottom',
               fontsize=10, fontweight='bold')

    ax.set_ylabel('Adoption Rate (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('LLM Application Areas (Illustrative)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()


def plot_chatbot_workflow():
    """Visualize chatbot workflow"""
    fig, ax = plt.subplots(figsize=(12, 8))

    steps = [
        ('Customer\nQuestion', 0.9, 'lightblue', '❓'),
        ('LLM\nProcessing', 0.7, 'lightcoral', '🤖'),
        ('Can\nAnswer?', 0.5, 'lightyellow', '🤔'),
        ('LLM\nResponse', 0.3, 'lightgreen', '💬'),
        ('Human\nAgent', 0.3, 'lightpink', '👤')
    ]

    # Main flow
    for i in range(4):
        text, y, color, emoji = steps[i]
        rect = plt.Rectangle((0.3, y - 0.06), 0.4, 0.12,
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.35, y, emoji, ha='left', va='center', fontsize=20)
        ax.text(0.5, y, text, ha='center', va='center',
               fontsize=11, fontweight='bold')

        if i < 3:
            ax.arrow(0.5, y - 0.07, 0, -0.06,
                    head_width=0.05, head_length=0.02,
                    fc='black', ec='black', linewidth=2)

    # Branch to human
    ax.arrow(0.5, 0.5, 0.15, -0.15,
            head_width=0.03, head_length=0.02,
            fc='red', ec='red', linewidth=2, linestyle='--')
    ax.text(0.6, 0.4, 'Complex\nCase', ha='left', fontsize=9,
           color='red', style='italic')

    # Human agent box
    text, y, color, emoji = steps[4]
    rect = plt.Rectangle((0.65, y - 0.06), 0.25, 0.12,
                        facecolor=color, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.67, y, emoji, ha='left', va='center', fontsize=20)
    ax.text(0.775, y, text, ha='center', va='center',
           fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('LLM Chatbot Workflow', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_code_assistance_uses():
    """Plot code assistance use cases"""
    uses = [
        'Code\nGeneration',
        'Bug\nFixing',
        'Code\nExplanation',
        'Test\nWriting',
        'Refactoring',
        'Documentation'
    ]

    usefulness = [90, 85, 95, 80, 75, 85]
    colors = ['lightcoral', 'lightblue', 'lightgreen',
             'lightyellow', 'lightpink', 'lavender']

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(uses, usefulness, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=2)

    # Add labels
    for bar, rate in zip(bars, usefulness):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
               f'{rate}%', ha='left', va='center',
               fontsize=11, fontweight='bold')

    ax.set_xlabel('Usefulness Rating (%)', fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_title('Code Assistance Use Cases', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_api_comparison():
    """Compare LLM API characteristics"""
    fig, ax = plt.subplots(figsize=(12, 7))

    apis = ['OpenAI\nGPT-4', 'Anthropic\nClaude', 'Google\nGemini', 'Open Source\nLlama']

    # Characteristics (illustrative, 0-10 scale)
    quality = [9, 9, 8, 7]
    cost = [3, 4, 5, 9]  # Higher = cheaper
    privacy = [5, 6, 5, 10]  # Higher = more private

    x = np.arange(len(apis))
    width = 0.25

    bars1 = ax.bar(x - width, quality, width, label='Quality', alpha=0.7,
                  color='lightblue', edgecolor='black')
    bars2 = ax.bar(x, cost, width, label='Cost Efficiency', alpha=0.7,
                  color='lightgreen', edgecolor='black')
    bars3 = ax.bar(x + width, privacy, width, label='Privacy', alpha=0.7,
                  color='lightcoral', edgecolor='black')

    ax.set_ylabel('Rating (0-10)', fontsize=12)
    ax.set_title('LLM API Comparison (Illustrative)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(apis)
    ax.legend()
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_ethical_considerations():
    """Visualize ethical considerations"""
    considerations = [
        'Bias &\nFairness',
        'Privacy &\nSecurity',
        'Truth &\nAccuracy',
        'Copyright &\nOwnership',
        'Job\nImpact'
    ]

    importance = [95, 90, 100, 75, 85]
    colors = ['lightcoral', 'lightyellow', 'lightblue', 'lightgreen', 'lightpink']

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(considerations, importance, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=2)

    # Add labels
    for bar, imp in zip(bars, importance):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
               f'{imp}%', ha='left', va='center',
               fontsize=11, fontweight='bold')

    ax.set_xlabel('Importance Rating (%)', fontsize=12)
    ax.set_xlim(0, 110)
    ax.set_title('Ethical Considerations in LLM Use',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_future_trends():
    """Plot future LLM trends"""
    trends = [
        'Multimodal',
        'Longer\nContext',
        'Specialized\nModels',
        'Better\nControl',
        'Energy\nEfficiency'
    ]

    timeline_years = [1, 2, 2, 3, 4]  # Years from now

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(trends, timeline_years, alpha=0.7,
                  color=['lightcoral', 'lightblue', 'lightgreen',
                        'lightyellow', 'lightpink'],
                  edgecolor='black', linewidth=2)

    # Add labels
    for bar, years in zip(bars, timeline_years):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
               f'{years} year{"s" if years > 1 else ""}',
               ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Expected Timeline (Years)', fontsize=12)
    ax.set_xlim(0, 5)
    ax.set_title('Future LLM Trends (Estimated)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
