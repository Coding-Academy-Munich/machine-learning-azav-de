from gettext import find
from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import warnings


def get_emoji_prop():
    """
    Returns a FontProperties object for a safe, vector-based emoji font.
    Prioritizes Monochrome fonts (NotoEmoji-Regular, Symbola) to avoid
    Matplotlib crashes with bitmap fonts (NotoColorEmoji).
    """
    # 1. List of safe, vector-based font files (in order of preference)
    #    We avoid 'NotoColorEmoji.ttf' because it causes 0x17 runtime errors.
    search_paths = [
        "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",  # Linux (Standard B&W)
        "/usr/share/fonts/truetype/noto-emoji/NotoEmoji-Regular.ttf",  # Linux (Alt path)
        "/usr/share/fonts/truetype/ancient-scripts/Symbola.ttf",  # Linux (Symbola)
        "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS (Works mostly)
        "C:\\Windows\\Fonts\\seguiemj.ttf",  # Windows
    ]

    selected_path = None

    # 2. Find the first existing file
    for path in search_paths:
        if os.path.exists(path):
            selected_path = path
            break

    # 3. Create FontProperties
    if selected_path:
        # We add the font file to the manager manually to ensure it's usable
        try:
            fm.fontManager.addfont(selected_path)
            prop = fm.FontProperties(fname=selected_path)
            return prop
        except Exception as e:
            warnings.warn(f"Found font at {selected_path} but failed to load: {e}")

    # 4. Fallback: Search installed system fonts by name (less reliable for files)
    safe_names = ["Noto Emoji", "Symbola", "Segoe UI Emoji", "Apple Color Emoji"]
    system_fonts = {f.name for f in fm.fontManager.ttflist}

    for name in safe_names:
        if name in system_fonts:
            return fm.FontProperties(family=name)

    warnings.warn("No specific emoji font found. Emojis may appear as squares.")
    return fm.FontProperties(family="DejaVu Sans")  # Default fallback


def plot_prompt_quality_spectrum():
    """Visualize spectrum of prompt quality"""
    fig, ax = plt.subplots(figsize=(14, 8))

    prompts = [
        ('Vague','Tell me\nabout AI', 0.2, "red"),
        ('Basic','Explain AI', 0.4, "orange"),
        ('Specific','Explain\nAI for\nbeginners', 0.6, "yellow"),
        ('Detailed','Explain\nAI for\nbeginners\nwith\nexamples', 0.8, "lightgreen"),
        (
            'Optimized','You are an\nexpert\nteacher.\nExplain AI\nto beginners\nwith 2\nreal-world\nexamples',
            1.0,
            "green",
        ),
    ]

    for header, text, x, color in prompts:
        # Circle
        circle = plt.Circle(
            (x, 0.72), 0.08, facecolor=color, edgecolor="black", linewidth=2
        )
        ax.add_patch(circle)

        # Text
        ax.text(
            x,
            0.55,
            header,
            ha="center",
            va="top",
            fontsize=24,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.5),
        )
        ax.text(
            x,
            0.45,
            text,
            ha="center",
            va="top",
            fontsize=24,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.5),
        )
        # Arrow
        if x < 1.0:
            ax.arrow(
                x + 0.08,
                0.7,
                0.03,
                0,
                head_width=0.02,
                head_length=0.01,
                fc="black",
                ec="black",
                linewidth=1.5,
            )

    # Axis
    # ax.plot([0.1, 1.1], [0.5, 0.5], "k-", linewidth=2, alpha=0.3)
    ax.text(
        0.1, 0.9, "Poor Quality", ha="left", fontsize=28, fontweight="bold", color="red"
    )
    ax.text(
        1.1,
        0.9,
        "High Quality",
        ha="right",
        fontsize=28,
        fontweight="bold",
        color="green",
    )

    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Prompt Quality Spectrum", fontsize=36, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()


def plot_shot_learning_comparison():
    """Compare zero-shot and few-shot learning"""
    emoji_prop = get_emoji_prop()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Zero-shot
    ax1.text(
        0.5,
        0.8,
        "Zero-Shot",
        ha="center",
        fontsize=36,
        fontweight="bold",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.6,
        "No examples provided",
        ha="center",
        fontsize=28,
        transform=ax1.transAxes,
        style="italic",
    )
    ax1.text(
        0.5,
        0.4,
        "❓ → 🤖 → ❓",
        ha="center",
        fontsize=40,
        fontproperties=emoji_prop,
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.2,
        "May work, but less reliable",
        ha="center",
        fontsize=28,
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.6),
    )
    ax1.axis("off")

    # Few-shot
    ax2.text(
        0.5,
        0.8,
        "Few-Shot",
        ha="center",
        fontsize=36,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.6,
        "2-5 examples provided",
        ha="center",
        fontsize=28,
        transform=ax2.transAxes,
        style="italic",
    )
    ax2.text(
        0.5,
        0.4,
        "✓ ✓ ✓ → 🤖 → ✓",
        ha="center",
        fontsize=40,
        fontproperties=emoji_prop,
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.2,
        "More accurate & consistent",
        ha="center",
        fontsize=28,
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.6),
    )
    ax2.axis("off")

    plt.suptitle("Learning Strategies", fontsize=48, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_chain_of_thought():
    """Visualize chain-of-thought prompting"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Steps
    steps = [
        ("Question", 0.9, "lightblue"),
        ("Step 1:\nBreak down", 0.7, "lightcoral"),
        ("Step 2:\nCalculate parts", 0.5, "lightgreen"),
        ("Step 3:\nCombine", 0.3, "lightyellow"),
        ("Final Answer", 0.1, "lightblue"),
    ]

    for text, y, color in steps:
        rect = plt.Rectangle(
            (0.25, y - 0.06), 0.5, 0.12, facecolor=color, edgecolor="black", linewidth=2
        )
        ax.add_patch(rect)
        ax.text(0.5, y, text, ha="center", va="center", fontsize=24, fontweight="bold")

        # Arrow
        if y > 0.15:
            ax.arrow(
                0.5,
                y - 0.07,
                0,
                -0.045,
                head_width=0.05,
                head_length=0.02,
                fc="black",
                ec="black",
                linewidth=2,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Chain-of-Thought: Think Step by Step", fontsize=36, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.show()


def plot_role_examples():
    """Visualize different role assignments"""

    # Get the font property object once
    emoji_prop = get_emoji_prop()

    roles = [
        ("  Teacher  \n👨‍🏫", "Simple\nExplanations"),
        ("    Expert   \n👨‍🔬", "Technical\nDetails"),
        (" Journalist \n📰", "Engaging\nStories"),
        (" Comedian \n🎭", "Humorous\nTone"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    x_positions = np.linspace(0.15, 0.85, len(roles))

    for i, (role, style) in enumerate(roles):
        x = x_positions[i]

        # Role - Pass the 'fontproperties' object explicitly
        ax.text(
            x,
            0.8,
            role,
            ha="center",
            va="center",
            fontsize=32,
            transform=ax.transAxes,
            fontproperties=emoji_prop,
            bbox=dict(
                boxstyle="round",
                facecolor="lightblue",
                edgecolor="black",
                linewidth=2,
                pad=0.4,
            ),
        )

        # Arrow
        ax.arrow(
            x,
            0.615,
            0,
            -0.22,
            transform=ax.transAxes,
            head_width=0.03,
            head_length=0.03,
            fc="black",
            ec="black",
            linewidth=1.5,
        )

        # Style
        ax.text(
            x,
            0.2,
            style,
            ha="center",
            va="center",
            fontsize=28,
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round",
                facecolor="lightyellow",
                edgecolor="black",
                linewidth=1,
                pad=0.5,
            ),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Role Assignment Changes Response Style", fontsize=36, fontweight="bold", pad=20
    )

    plt.tight_layout()
    plt.show()


def plot_prompt_template():
    """Visualize prompt template structure"""
    components = ["Role", "Task", "Context", "Constraints", "Format", "Examples"]

    colors = [
        "lightblue",
        "lightcoral",
        "lightgreen",
        "lightyellow",
        "lightpink",
        "lavender",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_start = 0.9
    y_step = 0.15

    for i, (component, color) in enumerate(zip(components, colors)):
        y = y_start - i * y_step

        rect = plt.Rectangle(
            (0.15, y - 0.07), 0.7, 0.14, facecolor=color, edgecolor="black", linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            0.5, y, component, ha="center", va="center", fontsize=24, fontweight="bold"
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Prompt Template Structure", fontsize=36, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()
