# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Modellvergleich</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Wann welches Modell?
#
# - Wir kennen jetzt lineare Regression und neuronale Netze
# - Wann sollten wir welches verwenden?
# - Schauen wir uns verschiedene Szenarien an

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from nn_training_deep_dive_plots import (
    plot_linear_comparison, plot_nonlinear_comparison,
    plot_small_data_comparison, plot_score_comparison
)

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Szenario 1: Lineare Beziehung
#
# - Perfekt für lineare Regression
# - Schauen wir, wie beide Modelle abschneiden

# %%
# Generate linear data
np.random.seed(42)
n_samples = 100
X_linear = np.random.uniform(0, 10, n_samples).reshape(-1, 1)
y_linear = 2 * X_linear.flatten() + 3 + np.random.normal(0, 1, n_samples)

# %%
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_linear, y_linear, test_size=0.3, random_state=42
)

# %%
# Train both models
lr_model = LinearRegression()
lr_model.fit(X_train_lin, y_train_lin)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42
)
mlp_model.fit(X_train_lin, y_train_lin)

# %%
# Generate predictions
x_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred_lr = lr_model.predict(x_plot)
y_pred_mlp = mlp_model.predict(x_plot)

# %% [markdown]
#
# ## Vergleich auf linearen Daten

# %%

# %%

# %% [markdown]
#
# ## Szenario 2: Nichtlineare Beziehung
#
# - Hier erwarten wir, dass das MLP besser ist
# - Lineare Regression wird Schwierigkeiten haben

# %%
# Generate non-linear data (quadratic)
X_nonlin = np.random.uniform(0, 10, n_samples).reshape(-1, 1)
y_nonlin = (X_nonlin.flatten() - 5) ** 2 + np.random.normal(0, 2, n_samples)

# %%
X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
    X_nonlin, y_nonlin, test_size=0.3, random_state=42
)

# %%
# Train both models on non-linear data
lr_model_nl = LinearRegression()
lr_model_nl.fit(X_train_nl, y_train_nl)

mlp_model_nl = MLPRegressor(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42
)
mlp_model_nl.fit(X_train_nl, y_train_nl)

# %%
# Generate predictions
y_pred_lr_nl = lr_model_nl.predict(x_plot)
y_pred_mlp_nl = mlp_model_nl.predict(x_plot)

# %% [markdown]
#
# ## Vergleich auf nichtlinearen Daten

# %%

# %%

# %% [markdown]
#
# ## Szenario 3: Wenig Daten
#
# - Was passiert mit sehr wenig Trainingsdaten?
# - Oft ist lineare Regression stabiler

# %%
# Generate small dataset (linear relationship)
n_small = 20
X_small = np.random.uniform(0, 10, n_small).reshape(-1, 1)
y_small = 2 * X_small.flatten() + 3 + np.random.normal(0, 1, n_small)

# %%
X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
    X_small, y_small, test_size=0.3, random_state=42
)

# %%
# Train both models on small dataset
lr_model_small = LinearRegression()
lr_model_small.fit(X_train_small, y_train_small)

mlp_model_small = MLPRegressor(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42
)
mlp_model_small.fit(X_train_small, y_train_small)

# %%
# Generate predictions
y_pred_lr_small = lr_model_small.predict(x_plot)
y_pred_mlp_small = mlp_model_small.predict(x_plot)

# %% [markdown]
#
# ## Vergleich mit wenig Daten

# %%

# %%

# %% [markdown]
#
# ## Vergleichstabelle

# %%
scenarios = ['Linear Data', 'Non-Linear Data', 'Small Dataset']
lr_test_scores = [
    lr_model.score(X_test_lin, y_test_lin),
    lr_model_nl.score(X_test_nl, y_test_nl),
    lr_model_small.score(X_test_small, y_test_small)
]
mlp_test_scores = [
    mlp_model.score(X_test_lin, y_test_lin),
    mlp_model_nl.score(X_test_nl, y_test_nl),
    mlp_model_small.score(X_test_small, y_test_small)
]

# %%

# %% [markdown]
#
# ## Vor- und Nachteile: Lineare Regression
#
# **Vorteile:**
# - Einfach und schnell
# - Gut interpretierbar
# - Stabil bei wenig Daten
# - Keine Hyperparameter-Optimierung nötig
#
# **Nachteile:**
# - Nur für lineare Beziehungen
# - Begrenzte Flexibilität

# %% [markdown]
#
# ## Vor- und Nachteile: Neuronale Netze (MLP)
#
# **Vorteile:**
# - Lernen komplexe, nichtlineare Muster
# - Sehr flexibel
# - Skalieren gut mit Datenmenge
#
# **Nachteile:**
# - Brauchen mehr Daten
# - Schwerer zu interpretieren ("Black Box")
# - Hyperparameter-Optimierung nötig
# - Rechenintensiver

# %% [markdown]
#
# ## Entscheidungshilfe
#
# **Verwende Lineare Regression, wenn:**
# - Die Beziehung linear ist (oder nahe dran)
# - Du wenig Daten hast
# - Interpretierbarkeit wichtig ist
# - Schnelligkeit wichtig ist
#
# **Verwende MLP, wenn:**
# - Die Beziehung komplex/nichtlinear ist
# - Du viele Daten hast
# - Vorhersagegenauigkeit am wichtigsten ist
# - Du bereit bist, Hyperparameter zu optimieren

# %% [markdown]
#
# ## In der Praxis
#
# - Oft: Starte mit einfachem Modell (lineare Regression)
# - Wenn Performance nicht gut genug → probiere komplexeres Modell
# - **Occam's Razor**: Bevorzuge einfachere Modelle, wenn sie gut genug sind
# - Mehr Komplexität ≠ Automatisch besser

# %% [markdown]
#
# ## Multivariate Regression
#
# - Bisher: Ein Feature (x)
# - Was ist mit mehreren Features?
# - Beide Modelle funktionieren mit mehreren Features

# %%
# Generate multivariate data with interaction
n_samples = 200
X_multi = np.random.uniform(0, 10, (n_samples, 2))
# Non-linear interaction: y depends on product of features
y_multi = (
    2 * X_multi[:, 0] +
    3 * X_multi[:, 1] +
    0.5 * X_multi[:, 0] * X_multi[:, 1] +  # Interaction term
    np.random.normal(0, 2, n_samples)
)

# %%
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42
)

# %%
# Train both models on multivariate data
lr_model_multi = LinearRegression()
lr_model_multi.fit(X_train_multi, y_train_multi)

mlp_model_multi = MLPRegressor(
    hidden_layer_sizes=(20, 10),
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
mlp_model_multi.fit(X_train_multi, y_train_multi)

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Lineare Regression**: Einfach, interpretierbar, gut für lineare Daten
# - **MLP**: Flexibel, mächtig, gut für komplexe Muster
# - Wahl hängt ab von:
#   - Art der Beziehung (linear vs. nichtlinear)
#   - Datenmenge
#   - Anforderungen (Genauigkeit vs. Interpretierbarkeit)
# - **Start simple, then complexify if needed**

# %% [markdown]
#
# ## Was kommt als Nächstes?
#
# - Wir haben die Grundlagen neuronaler Netze gelernt
# - Im nächsten Abschnitt: Workshop Zeit!
# - Praktische Übungen zur Vertiefung

# %%
