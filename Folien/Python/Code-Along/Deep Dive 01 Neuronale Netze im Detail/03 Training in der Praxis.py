# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Training in der Praxis</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Vom Konzept zur Praxis
#
# - Wir verstehen jetzt Gradientenabstieg
# - Aber wie setzt man das konkret um?
# - Schauen wir uns die praktischen Details an

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from nn_training_plots import (
    plot_training_data, plot_epochs_vs_score, plot_batch_sizes,
    plot_learning_rates_sgd_adam, plot_model_predictions,
    plot_overfitting, plot_regularization
)

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Epochen
#
# - Eine **Epoche** = einmal durch alle Trainingsdaten
# - Nach jeder Epoche: Modell hat alle Beispiele gesehen
# - Mehr Epochen = mehr Lernchancen
# - Aber: Irgendwann wird's nicht mehr besser

# %% [markdown]
#
# ## Beispieldaten vorbereiten

# %%
# Generate synthetic student data
np.random.seed(2025)
n_samples = 100
hours = np.random.uniform(0, 10, n_samples)
grades = 1.0 + 0.3 * hours + 0.05 * hours**2 + np.random.normal(0, 0.5, n_samples)
grades = np.clip(grades, 1, 6)

# %%
X = hours.reshape(-1, 1)
y = grades

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%

# %% [markdown]
#
# ## Training mit verschiedenen Epochenzahlen

# %%
epoch_values = [1, 5, 10, 50, 100, 500, 1000]
train_scores = []
test_scores = []

for epochs in epoch_values:
    mlp = MLPRegressor(
        hidden_layer_sizes=(10,), max_iter=epochs, random_state=42, solver="adam"
    )
    mlp.fit(X_train, y_train)

    train_scores.append(mlp.score(X_train, y_train))
    test_scores.append(mlp.score(X_test, y_test))

# %% [markdown]
#
# ## Score verbessert sich mit mehr Epochen

# %%

# %% [markdown]
#
# ## Batches und Mini-Batch Gradient Descent
#
# - In der Praxis: Nicht alle Daten auf einmal
# - Daten werden in **Batches** (Stapel) aufgeteilt
# - Gradient wird für jeden Batch berechnet
# - Vorteil: Schneller und speichereffizienter

# %% [markdown]
#
# ## Batch-Größen
#
# - **Batch Size** = Anzahl der Beispiele pro Batch
# - Kleine Batches (z.B. 32): Schneller, aber "verrauschter"
# - Große Batches (z.B. 256): Stabiler, aber langsamer
# - Typische Werte: 32, 64, 128, 256

# %% [markdown]
#
# ## Vergleich verschiedener Batch-Größen

# %%
batch_sizes = [1, 5, 10, 32, len(X_train)]  # Small, medium, full batch
batch_labels = ["Single(1)", "Small (5)", "Medium (10)", "Large (32)", "Full Batch"]
batch_scores = []

for batch_size in batch_sizes:
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500 // batch_size,
        batch_size=batch_size if batch_size <= len(X_train) else "auto",
        random_state=42,
        solver="sgd",
    )
    mlp.fit(X_train, y_train)
    batch_scores.append(mlp.score(X_test, y_test))

# %%


# %%

# %% [markdown]
#
# ## Die Lernrate in der Praxis
#
# - Erinnerung: Lernrate steuert die Schrittgröße
# - sklearn verwendet standardmäßig adaptive Lernraten
# - Der Adam-Optimierer passt die Lernrate automatisch an

# %% [markdown]
#
# ## Verschiedene Lernraten testen

# %%
learning_rates = [1e-5, 1e-4, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0]

# %%
lr_scores_sgd = []

for lr in learning_rates:
    mlp = MLPRegressor(
        hidden_layer_sizes=(10,),
        max_iter=200,
        learning_rate_init=lr,
        random_state=42,
        solver="sgd",
    )
    mlp.fit(X_train, y_train)
    lr_scores_sgd.append(mlp.score(X_test, y_test))

# %%
lr_scores_adam = []

for lr in learning_rates:
    mlp = MLPRegressor(
        hidden_layer_sizes=(10,),
        max_iter=200,
        learning_rate_init=lr,
        random_state=42,
        solver="adam",
    )
    mlp.fit(X_train, y_train)
    lr_scores_adam.append(mlp.score(X_test, y_test))


# %%

# %% [markdown]
#
# ## Overfitting: Zu viel des Guten
#
# - **Overfitting** = Das Modell lernt die Trainingsdaten auswendig
# - Es passt sich zu stark an Rauschen in den Daten an
# - Folge: Schlechte Performance auf neuen Daten

# %% [markdown]
#
# ## Demonstration: Overfitting

# %%
# Train models with different complexity (number of neurons)
neuron_counts = [2, 5, 20, 125, 200, 250]
n_train = 25
overfit_train_results = []
overfit_test_results = []
overfit_train_scores = []
overfit_test_scores = []


# %%
for n_neurons in neuron_counts:
    print(f"Training model with {n_neurons} neurons...", flush=True)
    mlp = MLPRegressor(
        hidden_layer_sizes=(n_neurons, n_neurons),
        max_iter=10_000,
        alpha=1e-8,
        random_state=42,
        solver="lbfgs",
    )
    mlp.fit(X_train[:n_train], y_train[:n_train])

    overfit_train_results.append(mlp.predict(X_train[:n_train]))
    overfit_test_results.append(mlp.predict(X_test))
    overfit_train_scores.append(mlp.score(X_train[:n_train], y_train[:n_train]))
    overfit_test_scores.append(mlp.score(X_test, y_test))

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Anzeichen von Overfitting
#
# - Training Score sehr hoch ✓
# - Test Score deutlich niedriger ✗
# - Große Lücke zwischen Training und Test
# - Das Modell ist zu komplex für die Daten

# %% [markdown]
#
# ## Wie vermeidet man Overfitting?
#
# - **Mehr Daten** sammeln
# - **Einfacheres Modell** verwenden (weniger Neuronen/Schichten)
# - **Regularisierung** anwenden
# - **Early Stopping**: Training stoppen, bevor Overfitting einsetzt

# %% [markdown]
#
# ## Early Stopping
#
# - Überwache Test-Performance während des Trainings
# - Stoppe, wenn Test-Performance nicht mehr besser wird
# - sklearn kann das automatisch machen

# %% [markdown]
#
# ## Early Stopping in sklearn

# %%

# %%

# %%

# %% [markdown]
#
# ## Regularisierung mit Alpha
#
# - **Regularisierung** bestraft zu komplexe Modelle
# - Der Parameter `alpha` steuert die Stärke
# - Höheres Alpha = stärkere Regularisierung = einfacheres Modell

# %%
alphas = [1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0]
reg_train_scores = []
reg_test_scores = []

for alpha in alphas:
    mlp = MLPRegressor(
        hidden_layer_sizes=(200, 200),
        max_iter=1_000,
        alpha=alpha,
        random_state=42,
        solver="lbfgs",
    )
    mlp.fit(X_train, y_train)

    reg_train_scores.append(mlp.score(X_train, y_train))
    reg_test_scores.append(mlp.score(X_test, y_test))


# %%

# %% [markdown]
#
# ## Praktische Tipps fürs Training
#
# 1. **Starte einfach**: Wenige Neuronen, wenige Schichten
# 2. **Überwache beide Scores**: Training und Test
# 3. **Verwende Early Stopping**: Verhindert Overfitting
# 4. **Experimentiere mit Hyperparametern**: Lernrate, Batch-Größe, Alpha
# 5. **Mehr Daten** helfen fast immer

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Epochen**: Wie oft durch die Daten
# - **Batches**: Daten in kleineren Portionen
# - **Lernrate**: Schrittgröße (oft automatisch angepasst)
# - **Overfitting**: Zu komplexes Modell für die Daten
# - **Early Stopping**: Rechtzeitig stoppen
# - **Regularisierung**: Komplexität einschränken

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - Vergleich verschiedener Modelle
# - Wann verwenden wir lineare Regression?
# - Wann verwenden wir neuronale Netze?

# %%
