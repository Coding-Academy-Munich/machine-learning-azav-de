# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: Erste Neuronale Netze</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Workshop: Erste Neuronale Netze (Optional)
#
# In diesem Workshop werden Sie:
# - Ihr erstes neuronales Netz trainieren
# - Mit verschiedenen Netzgrößen experimentieren
# - Neuronale Netze mit linearer Regression vergleichen
# - Sehen, wie Netze komplexe Probleme lösen

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# %%
sns.set_theme(style="darkgrid")
np.random.seed(42)

# %% [markdown]
#
# ## Aufgabe 1: Erstes neuronales Netz
#
# Erstellen Sie ein neuronales Netz, das eine quadratische Funktion lernt.
#
# 1. Die Daten sind bereits vorbereitet (siehe unten)
# 2. Erstellen Sie ein `MLPRegressor` mit 10 Neuronen
# 3. Trainieren Sie das Modell
# 4. Visualisieren Sie die Vorhersagen

# %%
# Generate quadratic data
X_quad = np.linspace(-3, 3, 100).reshape(-1, 1)
y_quad = X_quad.flatten() ** 2 + np.random.normal(0, 0.5, 100)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_quad, y_quad, test_size=0.3, random_state=42
)

# %% [markdown]
#
# ### Visualisierung der Daten

# %%
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.6, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.6, label='Test Data', color='orange')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
#
# ### Erstellen und Trainieren Sie das Modell
#
# - Verwenden Sie einen `MLPRegressor` mit einer versteckten Schicht und 10 Neuronen
# - Verwenden Sie den "lbfgs" Solver für bessere Konvergenz bei kleinen Datensätzen

# %%

# %%

# %% [markdown]
#
# ### Evaluieren Sie das Modell
#
# - Berechnen Sie den mean squared error auf Test-Daten
# - Berechnen Sie den R² Score auf Test-Daten

# %%

# %%

# %% [markdown]
#
# ### Visualisieren Sie die Vorhersagen
#
# - Erstellen Sie einen Plot mit den Daten und den Vorhersagen

# %%

# %% [markdown]
#
# ## Aufgabe 2: Netzgröße experimentieren
#
# Probieren Sie verschiedene Anzahlen von Neuronen aus:
#
# 1. Trainieren Sie Netze mit 5, 10 und 20 Neuronen
# 2. Vergleichen Sie die Performance (R² Score)
# 3. Welches funktioniert am besten?

# %%

# %% [markdown]
#
# ## Aufgabe 3: Vergleich mit linearer Regression
#
# Vergleichen Sie das neuronale Netz mit linearer Regression:
#
# 1. Trainieren Sie eine lineare Regression auf den gleichen Daten
# 2. Vergleichen Sie die R² Scores
# 3. Visualisieren Sie beide Vorhersagen
# 4. Was fällt Ihnen auf?

# %%

# %%

# %% [markdown]
#
# ## Aufgabe 4: Das XOR-Problem
#
# Lösen Sie das klassische XOR-Problem:
#
# 1. Die Daten sind bereits vorbereitet (siehe unten)
# 2. Erstellen Sie einen `MLPClassifier` mit 10 Neuronen
# 3. Trainieren Sie das Modell für 5000 Iterationen (`max_iter=5000`)
# 4. Prüfen Sie die Vorhersagen

# %%
# XOR data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# %% [markdown]
#
# ### XOR-Daten Visualisierung

# %%
plt.figure(figsize=(6, 6))
colors = ['blue' if label == 0 else 'red' for label in y_xor]
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200, alpha=0.6, edgecolors='black', linewidth=2)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('XOR Problem')
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()

# %% [markdown]
#
# ### Erstellen und Trainieren Sie das Modell
#
# - Verwenden Sie einen `MLPClassifier` mit zwei versteckten Schichten mit je 10 Neuronen
# - Setzen Sie `max_iter=5000` für ausreichende Trainingszeit

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Aufgabe 5: Halbmonde (Moons Dataset)
#
# Vergleichen Sie neuronale Netze mit linearen Modellen auf komplexen Daten:
#
# 1. Die "Moons" Daten sind bereits vorbereitet
# 2. Trainieren Sie einen `MLPClassifier` mit `(20, 5)` Neuronen
# 3. Trainieren Sie eine `LogisticRegression`
# 4. Vergleichen Sie die Accuracies
# 5. Was funktioniert besser?

# %%
# Generate moons dataset
X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42
)

# %% [markdown]
#
# ### Visualisierung der Moons Daten

# %%
plt.figure(figsize=(8, 6))
colors = ['red' if label == 0 else 'blue' for label in y_moons]
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Moons Dataset - Can a straight line separate these?')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
#
# ### Erstellen und Trainieren Sie das Modell

# %%

# %%

# %% [markdown]
#
# ### Erstellen und trainieren Sie ein lineares Modell

# %%

# %%

# %% [markdown]
#
# ### Vergleichen Sie die Genauigkeit

# %%

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# In diesem Workshop haben Sie:
# - ✓ Ihr erstes neuronales Netz erfolgreich trainiert
# - ✓ Mit verschiedenen Netzgrößen experimentiert
# - ✓ Gesehen, dass Netze besser sind als lineare Modelle bei nichtlinearen Daten
# - ✓ Das XOR-Problem gelöst (was linear unmöglich ist!)
# - ✓ Komplexe Muster in den Moons-Daten erkannt
#
# **Wichtigste Erkenntnis:**
# - Neuronale Netze können Probleme lösen, die für lineare Modelle unmöglich sind!
