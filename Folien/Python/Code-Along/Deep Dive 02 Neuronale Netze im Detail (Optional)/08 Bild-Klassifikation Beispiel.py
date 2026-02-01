# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Bild-Klassifikation Beispiel</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Unser erstes Bild-Klassifikations-Modell
#
# - Wir trainieren ein Modell zur Ziffernerkennung
# - Dataset: Handgeschriebene Ziffern (0-9)
# - Verwenden sklearn's MLPClassifier
# - Später: Vergleich mit CNNs

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Das Dataset laden

# %%
# Load digits dataset
digits = load_digits()


# %%

# %% [markdown]
#
# ## Beispielbilder anschauen

# %%
def plot_digit_examples(images, labels, n_examples=10):
    """Plot example digits from dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(n_examples):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}', fontsize=12)
        axes[i].axis('off')

    plt.suptitle('Example Handwritten Digits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Daten vorbereiten
#
# - Bilder sind 8×8 = 64 Pixel
# - Für MLP: Müssen sie "flachgeklopft" werden
# - Normalisierung: 0-16 → 0-1

# %%
# Flatten images to 1D arrays
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# %%

# %%
# Normalize to 0-1
X = X / 16.0

# %%

# %% [markdown]
#
# ## Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%

# %% [markdown]
#
# ## Modell erstellen und trainieren

# %%
# Create MLP classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42,
    verbose=True
)


# %%

# %% [markdown]
#
# ## Modell evaluieren

# %%

# %%

# %% [markdown]
#
# ## Detaillierter Classification Report

# %%

# %% [markdown]
#
# ## Confusion Matrix
#
# - Zeigt, welche Ziffern verwechselt werden
# - Diagonal: Richtige Vorhersagen
# - Außerhalb: Fehler

# %%
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.show()

# %%

# %% [markdown]
#
# ## Vorhersagen visualisieren

# %%
def plot_predictions(images, true_labels, pred_labels, n=20):
    """Plot predictions with correct/incorrect highlighting"""
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.ravel()

    for i in range(min(n, len(images))):
        axes[i].imshow(images[i].reshape(8, 8), cmap='gray')

        # Color: green if correct, red if wrong
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        axes[i].set_title(f'True: {true_labels[i]}\nPred: {pred_labels[i]}',
                         fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('Predictions (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Fehler genauer anschauen

# %%
# Find misclassified examples
errors = X_test[y_test != y_pred_test]
error_true = y_test[y_test != y_pred_test]
error_pred = y_pred_test[y_test != y_pred_test]


# %%

# %%
def plot_errors(errors, true_labels, pred_labels, n=10):
    """Plot misclassified examples"""
    n_show = min(n, len(errors))

    if n_show == 0:
        print("No errors to show!")
        return

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(n_show):
        axes[i].imshow(errors[i].reshape(8, 8), cmap='gray')
        axes[i].set_title(f'True: {true_labels[i]}\nPred: {pred_labels[i]}',
                         fontsize=10, color='red', fontweight='bold')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_show, 10):
        axes[i].axis('off')

    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Vorhersage-Wahrscheinlichkeiten
#
# - Das Modell gibt nicht nur eine Klasse aus
# - Es gibt **Wahrscheinlichkeiten** für jede Klasse
# - Wir können sehen, wie sicher das Modell ist

# %%
def plot_prediction_probabilities(images, true_labels, probabilities, n=None):
    """Plot prediction probabilities for examples"""
    if n is None:
        n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3*n))

    for i in range(n):
        # Show image
        axes[i, 0].imshow(images[i].reshape(8, 8), cmap='gray')
        axes[i, 0].set_title(f'True Label: {true_labels[i]}', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # Show probabilities
        axes[i, 1].bar(range(10), probabilities[i], alpha=0.7)
        axes[i, 1].set_xlabel('Digit', fontsize=10)
        axes[i, 1].set_ylabel('Probability', fontsize=10)
        axes[i, 1].set_title(f'Predicted: {np.argmax(probabilities[i])} '
                           f'(confidence: {np.max(probabilities[i]):.2%})',
                           fontsize=12)
        axes[i, 1].set_xticks(range(10))
        axes[i, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# %%
# Get probabilities for first few test examples
probabilities = mlp.predict_proba(X_test)

# %%

# %%

# %% [markdown]
#
# ## Was haben wir erreicht?
#
# - **~97% Genauigkeit** auf Testdaten
# - Das ist ziemlich gut!
# - Mit einem relativ einfachen MLP
# - Nur 8×8 Pixel

# %% [markdown]
#
# ## Vergleich: MLP vs CNN
#
# | Eigenschaft | MLP | CNN |
# |------------|-----|-----|
# | Input | Flattened (64 features) | 2D Image (8×8) |
# | Parameter | Viele | Weniger |
# | Positions-Invarianz | Nein | Ja |
# | Skaliert zu großen Bildern | Schlecht | Gut |
# | Genauigkeit auf MNIST | ~97% | ~99% |

# %% [markdown]
#
# ## Warum CNNs besser sind
#
# - **Nutzen die 2D-Struktur**: Berücksichtigen räumliche Nachbarschaft
# - **Positions-invariant**: Erkennen Muster überall im Bild
# - **Weniger Parameter**: Besonders bei größeren Bildern
# - **Hierarchisch**: Lernen von einfach zu komplex
# - **Skalieren besser**: Funktionieren gut auf 224×224 Bildern

# %% [markdown]
#
# ## Zusammenfassung
#
# - Wir haben ein **Bild-Klassifikations-Modell** trainiert
# - **~97% Genauigkeit** auf handgeschriebenen Ziffern
# - MLPs funktionieren, aber **CNNs sind besser** für Bilder
# - CNNs nutzen die Struktur von Bildern aus
# - Für echte Anwendungen: CNNs (z.B. mit TensorFlow/PyTorch)

# %% [markdown]
#
# ## Nächste Schritte
#
# - Workshop: Praktische Übungen mit Bildern
# - Später: Text-Verarbeitung und Large Language Models
# - Transfer Learning mit vortrainierten Modellen

# %%
