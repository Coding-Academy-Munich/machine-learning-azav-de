# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Richtung moderne NLP</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bisher gelernt haben
#
# - **Tokenisierung**: Text → Wörter
# - **Embeddings**: Wörter → dichte Vektoren
# - **Bag of Words / TF-IDF**: Einfache Text-Repräsentation
# - **Klassifikation**: Mit Standard ML-Modellen

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Das fehlende Stück: Reihenfolge
#
# - Bag of Words **ignoriert** die Reihenfolge
# - Aber Reihenfolge ist wichtig!
# - "The dog bit the man" ≠ "The man bit the dog"
# - "not good" ≠ "good"

# %% [markdown]
#
# ## Sequenzielle Modelle (Konzeptionell)
#
# - **Recurrent Neural Networks (RNNs)**
# - Verarbeiten Text Wort für Wort
# - Behalten "Gedächtnis" über bisherigen Text
# - Problem: Vergessen ältere Informationen

# %%
def plot_rnn_concept():
    """Conceptual visualization of RNN processing"""
    fig, ax = plt.subplots(figsize=(14, 6))

    words = ["The", "dog", "bit", "the", "man"]
    x_positions = np.arange(len(words)) * 2.5

    # Draw RNN cells
    for i, (word, x) in enumerate(zip(words, x_positions)):
        # Cell
        rect = plt.Rectangle((x, 1), 1.5, 1.2, facecolor='lightblue',
                            edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.75, 1.6, word, ha='center', va='center',
               fontsize=12, fontweight='bold')

        # Arrow to next cell (memory/hidden state)
        if i < len(words) - 1:
            ax.arrow(x + 1.6, 1.6, 0.7, 0,
                    head_width=0.2, head_length=0.1,
                    fc='red', ec='red', linewidth=2)
            ax.text(x + 2.0, 2.1, 'memory', ha='center',
                   fontsize=9, color='red', style='italic')

    ax.set_xlim(-0.5, x_positions[-1] + 2)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('RNN: Processing Text Sequentially (with Memory)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Das Aufmerksamkeits-Problem
#
# - Beim Lesen eines langen Textes:
# - Welche Wörter sind wichtig für das aktuelle Wort?
# - RNNs: Behandeln alle bisherigen Wörter gleich
# - Lösung: **Attention (Aufmerksamkeit)**

# %% [markdown]
#
# ## Attention-Mechanismus (Konzept)
#
# - **Idee**: Konzentriere dich auf relevante Wörter
# - Jedes Wort schaut auf alle anderen Wörter
# - Lernt automatisch, welche wichtig sind
# - Beispiel: Bei "Der König grüßt die Königin" ...
#   - "König" achtet auf "grüßt" und "Königin"

# %%
def plot_attention_concept():
    """Conceptual visualization of attention"""
    words = ["The", "king", "greets", "the", "queen"]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Position words in circle
    n = len(words)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Draw words
    for i, (word, xi, yi) in enumerate(zip(words, x, y)):
        circle = plt.Circle((xi, yi), 0.15, facecolor='lightblue',
                          edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(xi, yi, word, ha='center', va='center',
               fontsize=11, fontweight='bold')

    # Draw attention lines from "king" to other words
    king_idx = 1
    attention_weights = [0.1, 0.0, 0.6, 0.1, 0.7]  # Attention to each word

    for i in range(n):
        if i != king_idx and attention_weights[i] > 0:
            # Line width proportional to attention
            linewidth = attention_weights[i] * 5
            ax.plot([x[king_idx], x[i]], [y[king_idx], y[i]],
                   'r-', linewidth=linewidth, alpha=0.6)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Attention: "king" focuses on relevant words\n(Thicker lines = More attention)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Transformers: Die Revolution
#
# - **2017**: "Attention is All You Need" (Google)
# - Verwendet **nur** Attention (keine RNNs)
# - Viel schneller zu trainieren
# - Kann längere Texte verarbeiten
# - **Grundlage für alle modernen LLMs**

# %% [markdown]
#
# ## Der Weg zu LLMs
#
# 1. **Word Embeddings** (Word2Vec, GloVe)
# 2. **RNNs / LSTMs** (Sequenzielle Verarbeitung)
# 3. **Attention-Mechanismus**
# 4. **Transformers** (Nur Attention)
# 5. **BERT, GPT** (Große vortrainierte Transformers)
# 6. **ChatGPT, Claude** (Moderne LLMs)

# %%
def plot_nlp_evolution():
    """Plot evolution of NLP approaches"""
    approaches = [
        'Bag of\nWords',
        'Word\nEmbeddings',
        'RNNs/\nLSTMs',
        'Attention',
        'Transformers',
        'Modern\nLLMs'
    ]

    years = [2000, 2013, 2014, 2015, 2017, 2020]
    complexity = [1, 2, 3, 4, 5, 6]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot points and line
    ax.plot(years, complexity, 'o-', markersize=15, linewidth=3, color='steelblue')

    # Add labels
    for i, (year, comp, name) in enumerate(zip(years, complexity, approaches)):
        ax.text(year, comp + 0.3, name, ha='center', va='bottom',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax.set_xlabel('Year (approximate)', fontsize=13)
    ax.set_ylabel('Capability / Complexity', fontsize=13)
    ax.set_title('Evolution of NLP: From Bag of Words to LLMs',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 7)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Was macht LLMs besonders?
#
# - **Riesige Modelle**: Milliarden von Parametern
# - **Riesige Datenmengen**: Trainiert auf dem halben Internet
# - **Transfer Learning**: Ein Modell, viele Aufgaben
# - **Few-Shot Learning**: Lernt aus wenigen Beispielen
# - **Generierung**: Kann neue Texte erstellen

# %% [markdown]
#
# ## Größe der Modelle

# %%
def plot_model_sizes():
    """Plot comparison of model sizes"""
    models = ['Small\nNN\n(2010)', 'BERT\n(2018)', 'GPT-2\n(2019)', 'GPT-3\n(2020)', 'GPT-4\n(2023)']
    parameters = [1, 340, 1500, 175000, 1000000]  # In millions (GPT-4 estimated)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    bars = ax.bar(models, parameters, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, param in zip(bars, parameters):
        height = bar.get_height()
        if param >= 1000:
            label = f'{param/1000:.0f}B'
        else:
            label = f'{param}M'
        ax.text(bar.get_x() + bar.get_width()/2, height,
               label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Parameters (logarithmic scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Growth of Language Model Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - Von **Bag of Words** zu **Transformers**
# - **Attention**: Fokussiere auf relevante Wörter
# - **Transformers**: Nur Attention, sehr effizient
# - **LLMs**: Riesige Transformer-Modelle
# - Können **generieren**, **übersetzen**, **zusammenfassen**, etc.
# - Die Revolution der letzten Jahre!

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - **Large Language Models** im Detail
# - Wie funktionieren sie?
# - Was können sie?
# - Wie nutzt man sie?

# %%
