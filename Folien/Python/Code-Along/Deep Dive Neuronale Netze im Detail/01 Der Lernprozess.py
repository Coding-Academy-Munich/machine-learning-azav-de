# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Der Lernprozess</b>
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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from nn_training_plots import plot_training_error, plot_learning_stages

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
# ## Fehlermaße, die wir kennen
#
# - **MAE** (Mean Absolute Error): Durchschnitt der absoluten Fehler
# - **MSE** (Mean Squared Error): Durchschnitt der quadrierten Fehler
# - Bei neuronalen Netzen oft: MSE oder verwandte Maße

# %% [markdown]
#
# ## Beispiel: Notenvorhersage
#
# - Erinnerung: Wir sagen Noten basierend auf Lernstunden vorher
# - Schauen wir uns an, wie sich der Fehler während des Trainings ändert

# %%
# Generate student data
np.random.seed(42)
hours = np.linspace(0, 10, 50)
grades = 1.0 + 0.5 * hours + 0.05 * hours**2 + np.random.normal(0, 0.5, 50)
grades = np.clip(grades, 1, 6)

# %%
X = hours.reshape(-1, 1)
y = grades

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
#
# ## Training mit verschiedenen Epochen
#
# - Eine **Epoche** = einmal durch alle Trainingsdaten
# - Mehr Epochen = mehr Lerngelegenheiten
# - Aber: Irgendwann wird es nicht mehr besser

# %%
# Train models with different numbers of iterations
epoch_counts = [1, 5, 10, 50, 100, 500, 1000]
train_errors = []
test_errors = []

for max_iter in epoch_counts:
    mlp = MLPRegressor(
        hidden_layer_sizes=(10,),
        activation='relu',
        max_iter=max_iter,
        random_state=42,
        solver='sgd'
    )
    mlp.fit(X_train, y_train)

    train_pred = mlp.predict(X_train)
    test_pred = mlp.predict(X_test)

    train_mse = np.mean((y_train - train_pred) ** 2)
    test_mse = np.mean((y_test - test_pred) ** 2)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

# %% [markdown]
#
# ## Fehler während des Trainings

# %%

# %% [markdown]
#
# ## Was sehen wir?
#
# - Am Anfang: hoher Fehler (das Netz weiß noch nichts)
# - Während des Trainings: Fehler sinkt
# - Nach genug Training: Fehler stabilisiert sich
# - Das Netz hat gelernt!

# %% [markdown]
#
# ## Wie genau funktioniert das?
#
# - Das Netz hat Gewichte (weights) und Bias-Werte
# - Diese Parameter bestimmen die Vorhersagen
# - Beim Training werden diese Parameter angepasst
# - Ziel: Bessere Vorhersagen → kleinerer Fehler

# %% [markdown]
#
# ## Visualisierung: Vorhersagen verbessern sich

# %%
# Train models at different stages
x_plot = np.linspace(0, 10, 100).reshape(-1, 1)

# silence warnings for demonstration purposes
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    mlp_1 = MLPRegressor(hidden_layer_sizes=(500, 500), max_iter=1, random_state=42, solver='adam')
    mlp_1.fit(X_train, y_train)
    y_pred_1 = mlp_1.predict(x_plot)

    mlp_10 = MLPRegressor(hidden_layer_sizes=(500, 500), max_iter=10, random_state=42, solver='adam')
    mlp_10.fit(X_train, y_train)
    y_pred_10 = mlp_10.predict(x_plot)

    mlp_100 = MLPRegressor(hidden_layer_sizes=(500, 500), max_iter=100, random_state=42, solver='adam')
    mlp_100.fit(X_train, y_train)
    y_pred_100 = mlp_100.predict(x_plot)

    mlp_1000 = MLPRegressor(hidden_layer_sizes=(500, 500), max_iter=1000, random_state=42, solver='adam')
    mlp_1000.fit(X_train, y_train)
    y_pred_1000 = mlp_1000.predict(x_plot)


# %%

# %% [markdown]
#
# ## Die Loss-Funktion
#
# - Der Fehler wird auch **Loss** genannt
# - Die **Loss-Funktion** berechnet, wie schlecht das Modell ist
# - Für Regression: oft MSE (Mean Squared Error)
# - Für Klassifikation: andere Funktionen (z.B. Cross-Entropy)

# %% [markdown]
#
# ## Loss berechnen

# %%
def compute_mse_loss(y_true, y_pred):
    """Compute Mean Squared Error loss"""
    return np.mean((y_true - y_pred) ** 2)

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - Neuronale Netze lernen durch **Minimierung des Fehlers**
# - Der Fehler (Loss) wird mit einer **Loss-Funktion** berechnet
# - Mehr Training (Epochen) → kleinerer Fehler
# - Die **Parameter** (Gewichte und Bias) werden angepasst
# - Aber **wie** werden sie angepasst?
# - Das sehen wir in der nächsten Lektion!

# %%
