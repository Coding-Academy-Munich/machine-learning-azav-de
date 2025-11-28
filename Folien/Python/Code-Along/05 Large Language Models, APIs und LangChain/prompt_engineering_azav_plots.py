import numpy as np
import matplotlib.pyplot as plt


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
