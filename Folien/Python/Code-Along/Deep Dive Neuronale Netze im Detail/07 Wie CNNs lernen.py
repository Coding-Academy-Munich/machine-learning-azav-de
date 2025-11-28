# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Wie CNNs lernen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Die Magie der Schichten
#
# - CNNs haben mehrere Convolutional Layers
# - Jede Schicht lernt unterschiedliche Dinge
# - **Hierarchisches Lernen**: Von einfach zu komplex
# - Schauen wir uns das genauer an!

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Schicht 1: Einfache Kanten
#
# - Die **erste Schicht** lernt einfache Muster
# - Hauptsächlich: **Kanten** und **Linien**
# - Horizontal, vertikal, diagonal
# - Diese sind die "Bausteine" für komplexere Muster

# %% [markdown]
#
# ## Visualisierung: Erste-Schicht-Filter

# %%
# Create example filters that first layer might learn
first_layer_filters = {
    'Vertical Edge': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    'Horizontal Edge': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    'Diagonal \\': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'Diagonal /': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    'Corner': np.array([[-1, -1, 0], [-1, 2, 0], [0, 0, 0]]),
    'Blob': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
}

# %%
def plot_first_layer_filters():
    """Plot example filters from first layer"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for idx, (name, kernel) in enumerate(first_layer_filters.items()):
        # Normalize for visualization
        vmin, vmax = kernel.min(), kernel.max()
        im = axes[idx].imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')

        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle('First Layer Filters: Simple Edges', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Schicht 2: Formen und Texturen
#
# - Die **zweite Schicht** kombiniert Kanten
# - Lernt: **Ecken**, **Kurven**, **einfache Formen**
# - Auch: **Texturen** (z.B. gepunktet, gestreift)
# - Komplexer als Schicht 1

# %% [markdown]
#
# ## Konzeptuelle Visualisierung

# %%
def plot_layer_hierarchy():
    """Plot conceptual hierarchy of what each layer learns"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Layer 1: Edges
    layer1_text = ["Vertical\nEdges", "Horizontal\nEdges", "Diagonal\nEdges", "Colors"]
    axes[0].text(0.5, 0.9, 'Layer 1', ha='center', va='top',
                fontsize=16, fontweight='bold', transform=axes[0].transAxes)
    for i, text in enumerate(layer1_text):
        axes[0].text(0.25 + (i % 2) * 0.5, 0.7 - (i // 2) * 0.3, text,
                    ha='center', va='center', fontsize=11, transform=axes[0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[0].axis('off')

    # Layer 2: Shapes
    layer2_text = ["Corners", "Curves", "Simple\nShapes", "Textures"]
    axes[1].text(0.5, 0.9, 'Layer 2', ha='center', va='top',
                fontsize=16, fontweight='bold', transform=axes[1].transAxes)
    for i, text in enumerate(layer2_text):
        axes[1].text(0.25 + (i % 2) * 0.5, 0.7 - (i // 2) * 0.3, text,
                    ha='center', va='center', fontsize=11, transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    axes[1].axis('off')

    # Layer 3: Objects
    layer3_text = ["Eyes", "Wheels", "Doors", "Letters"]
    axes[2].text(0.5, 0.9, 'Layer 3+', ha='center', va='top',
                fontsize=16, fontweight='bold', transform=axes[2].transAxes)
    for i, text in enumerate(layer3_text):
        axes[2].text(0.25 + (i % 2) * 0.5, 0.7 - (i // 2) * 0.3, text,
                    ha='center', va='center', fontsize=11, transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    axes[2].axis('off')

    plt.suptitle('Hierarchical Learning in CNNs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Schicht 3+: Objektteile
#
# - **Tiefere Schichten** kombinieren Formen
# - Lernen **Objektteile**
# - Beispiele:
#   - Augen, Nasen (für Gesichter)
#   - Räder, Fenster (für Autos)
#   - Buchstaben, Ziffern (für Text)

# %% [markdown]
#
# ## Letzte Schichten: Ganze Objekte
#
# - Die **letzten Schichten** kombinieren alles
# - Erkennen **komplette Objekte**
# - Beispiele:
#   - Gesichter
#   - Autos
#   - Tiere
#   - Gebäude

# %% [markdown]
#
# ## Die Hierarchie im Überblick

# %%
def plot_cnn_hierarchy_flow():
    """Plot the flow from pixels to objects"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define layers with their learnings
    layers = [
        {'y': 5, 'name': 'Input Image', 'learns': ['Raw Pixels'], 'color': 'white'},
        {'y': 4, 'name': 'Conv Layer 1', 'learns': ['Edges', 'Lines', 'Colors'], 'color': 'lightblue'},
        {'y': 3, 'name': 'Conv Layer 2', 'learns': ['Corners', 'Curves', 'Textures'], 'color': 'lightcoral'},
        {'y': 2, 'name': 'Conv Layer 3', 'learns': ['Simple Parts', 'Patterns'], 'color': 'lightgreen'},
        {'y': 1, 'name': 'Dense Layers', 'learns': ['Complete Objects'], 'color': 'lightyellow'},
    ]

    for i, layer in enumerate(layers):
        # Draw layer box
        rect = plt.Rectangle((1, layer['y'] - 0.3), 12, 0.6,
                            facecolor=layer['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Layer name
        ax.text(1.5, layer['y'], layer['name'], fontsize=12, fontweight='bold',
               va='center', ha='left')

        # What it learns
        learns_text = ' | '.join(layer['learns'])
        ax.text(11.5, layer['y'], learns_text, fontsize=10,
               va='center', ha='right', style='italic')

        # Arrow to next layer
        if i < len(layers) - 1:
            ax.arrow(7, layer['y'] - 0.4, 0, -0.15,
                    head_width=0.3, head_length=0.05, fc='black', ec='black')

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('From Pixels to Objects: The CNN Hierarchy',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Warum funktioniert das?
#
# - **Komposition**: Komplexe Dinge aus einfachen Bausteinen
# - **Wiederverwendung**: Die gleiche Kante kann in vielen Objekten vorkommen
# - **Abstraktion**: Jede Schicht abstrahiert mehr
# - **Natürliche Hierarchie**: So funktioniert auch unsere visuelle Wahrnehmung!

# %% [markdown]
#
# ## Beispiel: Gesichtserkennung

# %%
def plot_face_recognition_hierarchy():
    """Plot how CNNs might learn to recognize faces"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Input
    axes[0].text(0.5, 0.5, '👤\nInput\nFace Image', ha='center', va='center',
                fontsize=14, transform=axes[0].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
    axes[0].set_title('Input', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Layer 1
    layer1_features = ['—\nHorizontal', '|\nVertical', '/\nDiagonal']
    for i, feat in enumerate(layer1_features):
        axes[1].text(0.5, 0.75 - i*0.3, feat, ha='center', va='center',
                    fontsize=10, transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[1].set_title('Layer 1: Edges', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Layer 2
    layer2_features = ['○\nCurves', '◢\nCorners', '▓\nTextures']
    for i, feat in enumerate(layer2_features):
        axes[2].text(0.5, 0.75 - i*0.3, feat, ha='center', va='center',
                    fontsize=10, transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    axes[2].set_title('Layer 2: Shapes', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Layer 3+
    layer3_features = ['👁\nEyes', '👃\nNose', '👄\nMouth']
    for i, feat in enumerate(layer3_features):
        axes[3].text(0.5, 0.75 - i*0.3, feat, ha='center', va='center',
                    fontsize=10, transform=axes[3].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    axes[3].set_title('Layer 3+: Face Parts', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.suptitle('Learning to Recognize Faces', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Transfer Learning
#
# - CNNs lernen **allgemeine Features**
# - Frühe Schichten (Kanten) sind für fast alle Bilder nützlich
# - Man kann ein trainiertes Netz nehmen und für neue Aufgaben anpassen
# - **Transfer Learning**: Nutze bereits gelernte Features

# %% [markdown]
#
# ## Transfer Learning Visualisierung

# %%
def plot_transfer_learning():
    """Visualize transfer learning concept"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pre-trained on ImageNet
    axes[0].text(0.5, 0.9, 'Pre-trained on ImageNet\n(1000 categories)', ha='center',
                fontsize=12, fontweight='bold', transform=axes[0].transAxes)

    layers_pretrained = [
        ('Conv Layers\n(Frozen)', 'lightblue', 0.6),
        ('Dense Layer\n(Frozen)', 'lightcoral', 0.3)
    ]

    for i, (name, color, y) in enumerate(layers_pretrained):
        rect = plt.Rectangle((0.2, y), 0.6, 0.15, facecolor=color,
                            edgecolor='black', linewidth=2, transform=axes[0].transAxes)
        axes[0].add_patch(rect)
        axes[0].text(0.5, y + 0.075, name, ha='center', va='center',
                    fontsize=10, transform=axes[0].transAxes)

    axes[0].axis('off')

    # Fine-tuned for new task
    axes[1].text(0.5, 0.9, 'Fine-tuned for Cats vs Dogs\n(2 categories)', ha='center',
                fontsize=12, fontweight='bold', transform=axes[1].transAxes)

    layers_finetuned = [
        ('Conv Layers\n(Frozen/Reused)', 'lightblue', 0.6),
        ('New Dense Layer\n(Trained)', 'lightgreen', 0.3)
    ]

    for i, (name, color, y) in enumerate(layers_finetuned):
        rect = plt.Rectangle((0.2, y), 0.6, 0.15, facecolor=color,
                            edgecolor='black', linewidth=2, transform=axes[1].transAxes)
        axes[1].add_patch(rect)
        axes[1].text(0.5, y + 0.075, name, ha='center', va='center',
                    fontsize=10, transform=axes[1].transAxes)

    axes[1].axis('off')

    plt.suptitle('Transfer Learning: Reuse Pre-trained Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Visualisierung von gelernten Features
#
# - Man kann visualisieren, was Filter in echten CNNs gelernt haben
# - Erste Schicht: Tatsächlich Kanten und Farben!
# - Tiefere Schichten: Komplexere Muster
# - Bestätigt unsere Theorie

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Hierarchisches Lernen**: Schicht für Schicht komplexer
# - **Schicht 1**: Kanten, Linien, Farben
# - **Schicht 2**: Ecken, Kurven, Texturen
# - **Schicht 3+**: Objektteile, dann ganze Objekte
# - **Transfer Learning**: Nutze gelernte Features für neue Aufgaben
# - **Natürlich**: Ähnlich wie menschliche Wahrnehmung

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - Praktisches Beispiel: MNIST-Ziffernerkennung
# - Training eines einfachen CNNs (mit sklearn)
# - Vergleich mit normalen Netzen

# %%
