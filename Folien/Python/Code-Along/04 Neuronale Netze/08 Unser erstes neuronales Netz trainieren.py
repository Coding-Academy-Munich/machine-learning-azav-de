# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Unser erstes neuronales Netz trainieren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Jetzt wird es praktisch!
#
# - Wir haben Neuronen und Netze kennengelernt
# - Jetzt trainieren wir unser erstes neuronales Netz
# - Wir verwenden unsere vertrauten Datensätze
# - Vergleichen mit linearer Regression

# %%
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Dataset: Studentennoten
#
# - Bekanntes Problem: Abschlussnote vorhersagen
# - Features: Zwischennote, Lernstunden, verpasste Stunden
# - Laden wir die Daten

# %% [markdown]
#
# Laden der Studentennoten

# %%
connection = sqlite3.connect("student_grades.db")
cursor = connection.cursor()

# %%
cursor.execute(
    "SELECT midterm_grade, study_hours, missed_classes, final_grade FROM grades"
)

# %%
rows = cursor.fetchall()

# %%
connection.close()

# %%
student_x = [list(row[:3]) for row in rows]
student_y = [row[3] for row in rows]

# %%

# %% [markdown]
#
# ## Train-Test Split
#
# - Wie immer: Daten aufteilen
# - 60% Training, 40% Test

# %%
student_x_train, student_x_test, student_y_train, student_y_test = train_test_split(
    student_x, student_y, test_size=0.4, random_state=42
)

# %%

# %% [markdown]
#
# ## Baseline: Lineare Regression
#
# - Erst trainieren wir ein lineares Modell
# - Das ist unser Vergleichswert

# %%
linear_model = LinearRegression()
linear_model.fit(student_x_train, student_y_train)

# %%

# %%

# %%
print("Linear Regression Performance:")
print(f"  Train MAE: {mean_absolute_error(student_y_train, linear_train_pred):.2f}")
print(f"  Test MAE:  {mean_absolute_error(student_y_test, linear_test_pred):.2f}")
print(f"  Train R²:  {r2_score(student_y_train, linear_train_pred):.3f}")
print(f"  Test R²:   {r2_score(student_y_test, linear_test_pred):.3f}")

# %% [markdown]
#
# ## Unser erstes neuronales Netz!
#
# - Verwenden wir `MLPRegressor` von sklearn
# - MLP = Multi-Layer Perceptron
# - Ähnliche API wie `LinearRegression`

# %% [markdown]
#
# ## Parameter für MLPRegressor
#
# - `hidden_layer_sizes`: Größe der versteckten Schichten
#   - `(10,)` = eine Schicht mit 10 Neuronen
#   - `(10, 5)` = zwei Schichten mit 10 und 5 Neuronen
# - `activation`: Aktivierungsfunktion (`'relu'` ist Standard)
# - `solver`: Optimierungsalgorithmus (z.B. `'lbfgs'`, `'adam'`, `'sgd'`)
# - `max_iter`: Maximale Anzahl von Trainingsiterationen (z.B. 500)
# - `random_state`: Für reproduzierbare Ergebnisse

# %%
# Create a simple neural network
mlp_model = MLPRegressor(
    hidden_layer_sizes=(10,),  # One hidden layer with 10 neurons
    activation='relu',          # ReLU activation function
    solver='lbfgs',            # LBFGS optimizer (efficient for small datasets)
    max_iter=500,              # Train for up to 500 iterations
    random_state=42            # For reproducibility
)

# %% [markdown]
#
# ## Trainieren des Netzes
#
# - Genau wie bei linearer Regression: `fit()`
# - Der Unterschied: Es dauert länger!
# - Das Netz muss viele Gewichte lernen

# %%

# %% [markdown]
#
# ## Vorhersagen machen

# %%

# %%

# %% [markdown]
#
# ## Vergleich der Ergebnisse

# %%
print("Linear Regression:")
print(f"  Train MAE: {mean_absolute_error(student_y_train, linear_train_pred):.2f}")
print(f"  Test MAE:  {mean_absolute_error(student_y_test, linear_test_pred):.2f}")
print(f"  Test R²:   {r2_score(student_y_test, linear_test_pred):.3f}")
print("\nNeural Network:")
print(f"  Train MAE: {mean_absolute_error(student_y_train, mlp_train_pred):.2f}")
print(f"  Test MAE:  {mean_absolute_error(student_y_test, mlp_test_pred):.2f}")
print(f"  Test R²:   {r2_score(student_y_test, mlp_test_pred):.3f}")

# %% [markdown]
#
# ## Interpretation
#
# - Beide Modelle schneiden ähnlich ab
# - Warum? Das Problem ist ziemlich linear!
# - Neuronale Netze zeigen ihre Stärke bei nichtlinearen Problemen

# %% [markdown]
#
# ## Visualisierung: Vorhersagen vs. Tatsächliche Werte

# %%
from building_nn_azav_plots import plot_predictions_comparison

# %%
plot_predictions_comparison(
    student_y_train, linear_train_pred,
    student_y_test, linear_test_pred,
    mlp_train_pred, mlp_test_pred
)

# %% [markdown]
#
# ## Experiment: Größeres Netzwerk
#
# - Was passiert mit mehr Neuronen?
# - Probieren wir 50 Neuronen

# %%
mlp_large = MLPRegressor(
    hidden_layer_sizes=(50,),
    activation='relu',
    solver='lbfgs',
    max_iter=500,
    random_state=42
)

# %%

# %%

# %%

# %% [markdown]
#
# ## Beobachtung
#
# - Mehr Neuronen ≠ immer besser
# - Bei wenig Trainingsdaten kann es sogar schlechter werden
# - **Overfitting**: Das Modell lernt die Trainingsdaten auswendig

# %% [markdown]
#
# ## Zusammenfassung
#
# - Wir haben unser erstes neuronales Netz trainiert!
# - sklearn macht es einfach: `MLPRegressor`
# - Bei linearen Problemen: ähnlich wie lineare Regression
# - Die wahre Stärke zeigt sich bei nichtlinearen Problemen
# - Vorsicht vor zu großen Netzen (Overfitting)

# %% [markdown]
#
# ## Workshop: Eiscreme-Verkauf mit neuronalen Netzen
#
# - Jetzt sind Sie dran!
# - Verwenden Sie den `ice_cream` Datensatz
# - Trainieren Sie ein neuronales Netz
# - Vergleichen Sie mit linearer Regression

# %% [markdown]
#
# ### Aufgabe 1: Daten laden
#
# - Laden Sie die Daten aus `ice_cream_sales.db`
# - Features: `temperature`, `cloud_cover`
# - Label: `sales`
# - Teilen Sie in Training und Test (60/40)

# %%
# Your code here: Load ice cream data
connection = sqlite3.connect("ice_cream_sales.db")
cursor = connection.cursor()

# %% [markdown]
#
# ### Aufgabe 2: Lineare Regression trainieren
#
# - Trainieren Sie ein lineares Modell
# - Berechnen Sie MAE und R² für Test-Daten

# %%
# Your code here: Train linear model


# %% [markdown]
#
# ### Aufgabe 3: Neuronales Netz trainieren
#
# - Erstellen Sie ein `MLPRegressor` mit 15 Neuronen
# - Trainieren Sie es
# - Vergleichen Sie die Ergebnisse mit linearer Regression

# %%
# Your code here: Train neural network


# %% [markdown]
#
# ### Aufgabe 4: Visualisierung
#
# - Plotten Sie Vorhersagen vs. tatsächliche Werte
# - Für beide Modelle

# %%
# Your code here: Create comparison plot


# %% [markdown]
#
# ### Bonus-Aufgabe: Experimentieren
#
# - Probieren Sie verschiedene Netzwerk-Größen
# - Was passiert mit 5 Neuronen? Mit 100?
# - Probieren Sie zwei Schichten: `(20, 10)`

# %%
# Your code here: Experiment with different architectures

