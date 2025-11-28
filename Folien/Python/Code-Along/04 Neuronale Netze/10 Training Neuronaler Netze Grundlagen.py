# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Training Neuronaler Netze: Grundlagen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Rückblick: Unser erstes Netzwerk
#
# - Im letzten Thema haben wir ein neuronales Netz trainiert
# - Wir haben `mlp_model.fit(X_train, y_train)` aufgerufen
# - Das Netz hat gelernt, Vorhersagen zu machen
# - Aber **wie** lernt es eigentlich?

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Das Lernziel
#
# - Beim Training hat das Netz ein klares Ziel
# - **Minimiere den Fehler** bei den Vorhersagen
# - Der Fehler misst, wie weit die Vorhersagen von den echten Werten entfernt sind

# %% [markdown]
#
# ## Die Loss-Funktion
#
# - Der Fehler wird auch **Loss** genannt
# - Die **Loss-Funktion** berechnet, wie schlecht das Modell ist
# - Für Regression: oft MSE (Mean Squared Error)
# - Für Klassifikation: Cross-Entropy
# - **Ziel des Trainings**: Loss minimieren!

# %% [markdown]
#
# ## Epochen: Wie oft lernen wir?
#
# - Eine **Epoche** = einmal durch alle Trainingsdaten
# - Mehr Epochen = mehr Lerngelegenheiten
# - Aber: Irgendwann wird es nicht mehr besser
# - **Zu wenig**: Netz hat nicht genug gelernt (underfitting)
# - **Zu viel**: Netz lernt Trainingsdaten auswendig (overfitting)

# %% [markdown]
#
# Erzeugen von Beispieldaten

# %%
np.random.seed(42)

# %%
X, y = make_moons(n_samples=100, noise=0.2)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
#
# Trainieren mit verschiedenen Epochen-Anzahlen

# %%
epoch_list = [1, 10, 50, 200, 1000]

# %%
train_scores = []

# %%
test_scores = []

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for max_iter in epoch_list:
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=max_iter, random_state=42)
        mlp.fit(X_train, y_train)
        train_scores.append(mlp.score(X_train, y_train))
        test_scores.append(mlp.score(X_test, y_test))

# %%

# %% [markdown]
#
# ## Lernrate (Learning Rate)
#
# - Wie groß sind die Anpassungsschritte?
# - **Zu klein**: Lernen dauert ewig
# - **Zu groß**: Das Netz "springt" über die Lösung
# - **Goldlöckchen-Prinzip**: Nicht zu groß, nicht zu klein!

# %% [markdown]
#
# ## Visualisierung: Lernrate
#
# - Stellen Sie sich vor, Sie suchen den tiefsten Punkt in einem Tal
# - **Kleine Lernrate**: Kleine Schritte - sicher aber langsam
# - **Große Lernrate**: Große Schritte - schnell aber riskant
# - **Richtige Lernrate**: Gute Balance zwischen Geschwindigkeit und Präzision

# %%
# Compare different learning rates (using SGD solver)
learning_rates = [0.001, 0.01, 0.1]
colors = ['blue', 'green', 'red']

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    plt.figure(figsize=(10, 5))
    for lr, color in zip(learning_rates, colors):
        mlp = MLPClassifier(
            hidden_layer_sizes=(10,),
            max_iter=100,
            learning_rate_init=lr,
            solver='sgd',
            random_state=42
        )
        mlp.fit(X_train, y_train)

        # Plot loss curve
        plt.plot(mlp.loss_curve_, label=f'Learning Rate = {lr}', color=color, linewidth=2)

    plt.xlabel('Epochen / Epochs')
    plt.ylabel('Loss')
    plt.title('Einfluss der Lernrate / Impact of Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
#
# ## Batch Size
#
# - **Batch Size**: Wie viele Beispiele auf einmal verarbeiten?
# - **Vollständiger Batch**: Alle Daten auf einmal - langsam aber stabil
# - **Mini-Batch**: Kleine Gruppen - schneller, etwas rauschiger
# - **Einzelnes Beispiel (SGD)**: Sehr schnell, sehr rauschig
# - Üblich: Mini-Batches von 32, 64, 128 Beispielen

# %% [markdown]
#
# ## Early Stopping
#
# - Problem: Woher wissen wir, wann wir aufhören sollen?
# - **Early Stopping**: Stoppe, wenn Test-Fehler nicht mehr besser wird
# - Verhindert Overfitting
# - sklearn macht das automatisch (wenn `early_stopping=True`)

# %%
# Example with early stopping
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    mlp_early = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
    mlp_early.fit(X_train, y_train)

# %%

# %% [markdown]
#
# ## Regularisierung: Overfitting vermeiden
#
# - **Regularisierung**: Verhindert zu komplexe Modelle
# - Parameter: `alpha` (höher = stärkere Regularisierung)
# - **Zu klein (`alpha=0`)**: Overfitting möglich
# - **Zu groß**: Underfitting
# - Standard: `alpha=0.0001` (funktioniert oft gut)

# %%
# Compare different alpha values
alphas = [0.0, 0.0001, 0.01, 1.0]

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    plt.figure(figsize=(12, 5))
    for alpha in alphas:
        mlp = MLPClassifier(
            hidden_layer_sizes=(100,),
            alpha=alpha,
            max_iter=500,
            random_state=42
        )
        mlp.fit(X_train, y_train)

        train_acc = mlp.score(X_train, y_train)
        test_acc = mlp.score(X_test, y_test)

        plt.bar([f'α={alpha}\nTrain', f'α={alpha}\nTest'],
                [train_acc, test_acc],
                alpha=0.7)

    plt.ylabel('Genauigkeit / Accuracy')
    plt.title('Einfluss von Regularisierung (alpha) / Impact of Regularization (alpha)')
    plt.ylim([0.5, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# %% [markdown]
#
# ## Praktische Tipps
#
# ### Anzahl Epochen
# - Start: 100-200 Epochen
# - Nutze `early_stopping=True` für automatisches Stoppen
#
# ### Lernrate
# - sklearn wählt meist gute Standardwerte
# - Bei Problemen: Standardwerte aus sklearn Dokumentation prüfen
#
# ### Regularisierung
# - Start mit `alpha=0.0001` (sklearn Standard)
# - Bei Overfitting: `alpha` erhöhen (z.B. 0.001, 0.01)
# - Bei Underfitting: `alpha` verringern (z.B. 0.00001)

# %% [markdown]
#
# ## Wann ist mein Modell überangepasst?
#
# - **Training-Fehler** ist sehr klein
# - **Test-Fehler** ist deutlich größer
# - Das Netz hat die Trainingsdaten auswendig gelernt
# - **Lösung**: Mehr Daten, Regularisierung, oder Early Stopping

# %%
# Demonstration of overfitting
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Overfit model (too complex, too many iterations, no regularization)
    mlp_overfit = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        alpha=0.0,  # No regularization
        max_iter=1000,
        random_state=42
    )
    mlp_overfit.fit(X_train, y_train)

    # Good model (balanced)
    mlp_good = MLPClassifier(
        hidden_layer_sizes=(10,),
        alpha=0.001,  # Some regularization
        max_iter=200,
        early_stopping=True,
        random_state=42
    )
    mlp_good.fit(X_train, y_train)

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Loss-Funktion**: Misst, wie schlecht das Modell ist
# - **Epochen**: Wie oft durch die Daten (zu viele → Overfitting)
# - **Lernrate**: Schrittgröße beim Lernen (Goldlöckchen-Prinzip!)
# - **Batch Size**: Wie viele Beispiele gleichzeitig
# - **Early Stopping**: Automatisch stoppen bei Overfitting
# - **Regularisierung (alpha)**: Verhindert zu komplexe Modelle
#
# **Wichtigste Regel**: Trainings- UND Test-Fehler beobachten!

# %%
