# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Convolutional Neural Networks</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Das Problem mit normalen Netzen
#
# - Bilder haben sehr viele Pixel (Features)
# - 28×28 Bild = 784 Features
# - 224×224×3 Bild = 150.528 Features!
# - Standard-Netze (MLP) brauchen extrem viele Parameter
# - Wir brauchen eine bessere Lösung

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from scipy import signal

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Die Lösung: Convolutional Neural Networks (CNNs)
#
# - **Spezielle Netze für Bilder**
# - Nutzen die Struktur von Bildern aus
# - Weniger Parameter als normale Netze
# - Erkennen **lokale Muster**

# %% [markdown]
#
# ## Kernidee: Filter (Kernel)
#
# - Ein **Filter** ist ein kleines Muster (z.B. 3×3)
# - Dieser Filter wird über das Bild geschoben
# - An jeder Position: Berechne eine Zahl
# - Das Ergebnis: Eine "Feature Map"

# %% [markdown]
#
# ## Beispiel-Filter: Kanten-Detektor

# %%
# Create a simple vertical edge detector filter
vertical_edge_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# %%
# Create a horizontal edge detector filter
horizontal_edge_filter = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# %%

# %% [markdown]
#
# ## Wie funktioniert ein Filter?
#
# - Filter wird über Bild geschoben
# - An jeder Position: **Element-weise Multiplikation**
# - Ergebnisse werden summiert
# - Das ergibt einen Wert in der Feature Map

# %% [markdown]
#
# ## Visualisierung der Faltung (Convolution)

# %%
# Load a digit for demonstration
digits = load_digits()
sample_digit = digits.images[0]

# %%
def apply_convolution(image, kernel):
    """Apply convolution to image"""
    # Use scipy's convolve2d for demonstration
    return signal.convolve2d(image, kernel, mode='valid')

# %%
# Apply filters
vertical_features = apply_convolution(sample_digit, vertical_edge_filter)
horizontal_features = apply_convolution(sample_digit, horizontal_edge_filter)

# %%
def plot_filter_application(original, filtered_v, filtered_h):
    """Plot original image and filtered results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Vertical edges
    axes[1].imshow(vertical_features, cmap='gray')
    axes[1].set_title('Vertical Edges Detected')
    axes[1].axis('off')

    # Horizontal edges
    axes[2].imshow(horizontal_features, cmap='gray')
    axes[2].set_title('Horizontal Edges Detected')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Was passiert hier?
#
# - **Vertikaler Filter**: Erkennt vertikale Kanten
# - **Horizontaler Filter**: Erkennt horizontale Kanten
# - Jeder Filter spezialisiert sich auf ein Muster
# - CNNs **lernen** diese Filter automatisch!

# %% [markdown]
#
# ## Mehrere Filter

# %%
# Create more filters
diagonal_filter_1 = np.array([
    [-1, 0, 0],
    [ 0, 1, 0],
    [ 0, 0, 1]
])

diagonal_filter_2 = np.array([
    [ 0, 0, -1],
    [ 0, 1,  0],
    [ 1, 0,  0]
])

blur_filter = np.ones((3, 3)) / 9

# %%
# Apply all filters
diag1_features = apply_convolution(sample_digit, diagonal_filter_1)
diag2_features = apply_convolution(sample_digit, diagonal_filter_2)
blur_features = apply_convolution(sample_digit, blur_filter)

# %%
def plot_multiple_filters(original):
    """Plot results of multiple filters"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original (twice)
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Vertical edges
    axes[0, 1].imshow(vertical_features, cmap='gray')
    axes[0, 1].set_title('Vertical Edges')
    axes[0, 1].axis('off')

    # Horizontal edges
    axes[0, 2].imshow(horizontal_features, cmap='gray')
    axes[0, 2].set_title('Horizontal Edges')
    axes[0, 2].axis('off')

    # Diagonal 1
    axes[1, 0].imshow(diag1_features, cmap='gray')
    axes[1, 0].set_title('Diagonal Edges (\\)')
    axes[1, 0].axis('off')

    # Diagonal 2
    axes[1, 1].imshow(diag2_features, cmap='gray')
    axes[1, 1].set_title('Diagonal Edges (/)')
    axes[1, 1].axis('off')

    # Blur
    axes[1, 2].imshow(blur_features, cmap='gray')
    axes[1, 2].set_title('Blur (Average)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Pooling: Größe reduzieren
#
# - Nach Convolution: Bild ist immer noch groß
# - **Pooling** reduziert die Größe
# - Typisch: **Max Pooling**
# - Nimmt das Maximum aus einem kleinen Bereich (z.B. 2×2)

# %% [markdown]
#
# ## Max Pooling Visualisierung

# %%
def max_pool_2d(image, pool_size=2):
    """Apply max pooling to image"""
    h, w = image.shape
    new_h = h // pool_size
    new_w = w // pool_size

    pooled = np.zeros((new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            pooled[i, j] = np.max(
                image[i*pool_size:(i+1)*pool_size,
                      j*pool_size:(j+1)*pool_size]
            )

    return pooled

# %%
# Apply max pooling
pooled_digit = max_pool_2d(sample_digit, pool_size=2)

# %%
def plot_pooling_effect(original, pooled):
    """Plot original and pooled images"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original ({original.shape[0]}×{original.shape[1]})')
    axes[0].axis('off')

    # Pooled
    axes[1].imshow(pooled, cmap='gray')
    axes[1].set_title(f'After Max Pooling ({pooled.shape[0]}×{pooled.shape[1]})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Warum Pooling?
#
# - Reduziert die Anzahl der Parameter
# - Macht das Netz robuster gegenüber kleinen Verschiebungen
# - Behält die wichtigsten Informationen
# - Schnellere Berechnungen

# %% [markdown]
#
# ## Aufbau eines CNN
#
# 1. **Input**: Bild
# 2. **Convolutional Layer**: Mehrere Filter anwenden
# 3. **Activation**: ReLU (wie bei normalen Netzen)
# 4. **Pooling**: Größe reduzieren
# 5. Wiederhole Schritte 2-4 mehrmals
# 6. **Flatten**: Zu 1D-Array machen
# 7. **Dense Layers**: Normale neuronale Netz-Schichten
# 8. **Output**: Klassifikation

# %% [markdown]
#
# ## Visualisierung: CNN-Architektur

# %%
def plot_cnn_architecture():
    """Plot simplified CNN architecture"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Layer sizes for visualization
    layers = [
        {'name': 'Input\n28×28×1', 'x': 1, 'size': 28, 'color': 'lightblue'},
        {'name': 'Conv\n26×26×32', 'x': 3, 'size': 26, 'color': 'lightcoral'},
        {'name': 'Pool\n13×13×32', 'x': 5, 'size': 13, 'color': 'lightgreen'},
        {'name': 'Conv\n11×11×64', 'x': 7, 'size': 11, 'color': 'lightcoral'},
        {'name': 'Pool\n5×5×64', 'x': 9, 'size': 5, 'color': 'lightgreen'},
        {'name': 'Flatten\n1600', 'x': 11, 'size': 3, 'color': 'lightyellow'},
        {'name': 'Dense\n10', 'x': 13, 'size': 1, 'color': 'lightgray'}
    ]

    for layer in layers:
        # Draw rectangle representing layer
        height = layer['size'] / 3  # Scale for visualization
        rect = plt.Rectangle((layer['x'], 2.5 - height/2), 0.8, height,
                            facecolor=layer['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Add layer name
        ax.text(layer['x'] + 0.4, 1, layer['name'],
               ha='center', va='top', fontsize=10, fontweight='bold')

        # Draw arrows between layers
        if layer['x'] < 13:
            ax.arrow(layer['x'] + 0.9, 2.5, 0.9, 0,
                    head_width=0.2, head_length=0.1, fc='black', ec='black')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('CNN Architecture (Simplified)', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Parameter-Effizienz
#
# - **Normales Netz** auf 28×28 Bild:
#   - Input: 784 Features
#   - Hidden Layer (100 Neuronen): 78.400 Parameter!
#
# - **CNN** auf 28×28 Bild:
#   - Ein 3×3 Filter: Nur 9 Parameter
#   - 32 Filter: 32 × 9 = 288 Parameter
#   - Viel effizienter!

# %% [markdown]
#
# ## Vorteile von CNNs
#
# - **Weniger Parameter** als normale Netze
# - **Erkennen lokale Muster** (Kanten, Texturen, etc.)
# - **Positions-invariant**: Erkennen Muster überall im Bild
# - **Hierarchisch**: Erste Layer → einfache Muster, tiefe Layer → komplexe Muster
# - **Sehr erfolgreich** für Bildverarbeitung

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Convolution**: Filter werden über Bild geschoben
# - **Filter/Kernel**: Kleine Muster (z.B. 3×3), die Features erkennen
# - **Feature Maps**: Ergebnis der Convolution
# - **Pooling**: Reduziert Größe, behält wichtige Info
# - **CNN-Architektur**: Conv → Pool → Conv → Pool → Dense
# - **Vorteil**: Viel weniger Parameter als normale Netze

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - Wie CNNs Schicht für Schicht lernen
# - Von Kanten über Formen zu Objekten
# - Visualisierung dessen, was CNNs "sehen"

# %%
