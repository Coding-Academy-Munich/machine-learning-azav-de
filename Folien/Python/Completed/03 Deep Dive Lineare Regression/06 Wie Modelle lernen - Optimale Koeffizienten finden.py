# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Wie Modelle lernen - Optimale Koeffizienten finden</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Das Lernproblem
#
# Gegeben:
# - Trainingsdaten X (Features)
# - Zielwerte y (was wir vorhersagen wollen)
#
# Gesucht:
# - Koeffizienten und Intercept
# - Die die Fehler minimieren

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %% [markdown]
#
# ## Ein einfaches Beispiel

# %% [markdown]
#
# Einfache Trainingsdaten

# %%
X_simple = np.array([[1], [2], [3], [4], [5]])
y_simple = np.array([3, 5, 7, 9, 11])

# %% [markdown]
#
# Visualisiere

# %%
plt.figure(figsize=(8, 5))
plt.scatter(X_simple, y_simple, s=100, alpha=0.7)
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.title("Training Data")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
#
# ## Was sind "gute" Koeffizienten?
#
# Wir wollen eine Linie, die:
# - Möglichst nah an allen Punkten ist
# - Die Fehler minimiert
# - Gut generalisiert

# %% [markdown]
#
# Probiere verschiedene Koeffizienten

# %%
from plot_utils import plot_coefficients_effect

# %%
coefficients_to_try = [1.5, 2.0, 2.5]
intercepts_to_try = [1.5, 1.0, 0.5]

# %%
plot_coefficients_effect(X_simple, y_simple, coefficients_to_try, intercepts_to_try)
# %% [markdown]
#
# ## Die Kosten von Fehlern
#
# **Mean Squared Error (MSE)**:
# - Misst durchschnittliche quadratische Abweichung
# - Bestraft große Fehler stärker
# - MSE = $(1/n) × \sum(y_{\textrm{true}} - y_{\textrm{pred}})²$

# %% [markdown]
#
# ## Visualisierung der Fehleroberfläche

# %% [markdown]
#
# Plotte Fehleroberfläche
#
# - 3D-Oberfläche
# - Konturplot

# %%
from plot_utils import plot_error_surface

# %%
best_coef, best_intercept, min_mse = plot_error_surface(X_simple, y_simple)

# %%
print(f"Best coefficient: {best_coef:.3f}")
print(f"Best intercept: {best_intercept:.3f}")
print(f"Minimum MSE: {min_mse:.3f}")

# %% [markdown]
#
# ## Die mathematische Lösung
#
# Es gibt eine Formel für die optimalen Koeffizienten!
#
# **Normalgleichung** (vereinfacht):
# - Koeffizienten = $(X^T × X)^{-1} × X^T × y$
# - $X^T$ = Transponierte von $X$
# - $M^{-1}$ = Inverse von $M$

# %% [markdown]
#
# ## Warum funktioniert das?
#
# - Die Normalgleichung findet das Minimum der Fehleroberfläche
# - Mathematisch: Setzt die Ableitung der Kostenfunktion auf 0
# - Resultat: Optimale Koeffizienten in einem Schritt!
#
# (Die Herleitung ist komplex, aber die Anwendung ist einfach!)

# %% [markdown]
#
# ## Schritt 1: Daten vorbereiten
#
# Für den Intercept fügen wir eine Spalte mit Einsen hinzu:

# %%
X_simple

# %%
X_with_intercept = np.column_stack([np.ones(len(X_simple)), X_simple])

# %%
X_with_intercept

# %% [markdown]
#
# ## Schritt 2: Normalgleichung anwenden

# %% [markdown]
#
# Normalgleichung: $θ = (X^T X)^{-1} X^T y$

# %%
X = X_with_intercept

# %%
X_transpose = X.T

# %%
X_transpose

# %% [markdown]
#
# Berechne $X^T × X$

# %%
XtX = X_transpose @ X

# %%
XtX

# %% [markdown]
#
# Berechne Inverse

# %%
XtX_inv = np.linalg.inv(XtX)

# %%
XtX_inv

# %%
XtX_inv @ XtX

# %%
XtX @ XtX_inv

# %% [markdown]
#
# Berechne X^T × y

# %%
Xty = X_transpose @ y_simple

# %%
Xty

# %% [markdown]
#
# Finale Berechnung: $θ = (X^T × X)^{-1} × X^T × y$

# %%
theta = XtX_inv @ Xty

# %% [markdown]
#
# Optimale Parameter:
# - Intercept (Y-Abschnitt): $\theta[0]$
# - Koeffizient: $\theta[1]$

# %%
theta[0]

# %%
theta[1]

# %% [markdown]
#
# ## Vergleich mit Scikit-Learn

# %% [markdown]
#
# Trainiere Scikit-Learn-Modell

# %%
sklearn_model = LinearRegression()
sklearn_model.fit(X_simple, y_simple)

# %%
print("Our solution:")
print(f"  Intercept: {theta[0]:.3f}")
print(f"  Coefficient: {theta[1]:.3f}")

print("\nScikit-Learn:")
print(f"  Intercept: {sklearn_model.intercept_:.3f}")
print(f"  Coefficient: {sklearn_model.coef_[0]:.3f}")

if np.isclose(theta[0], sklearn_model.intercept_) and np.isclose(
    theta[1], sklearn_model.coef_[0]
):
    print("\nThey match! ✓")
else:
    print("\nUh, oh! They do not match! ✗")

# %% [markdown]
#
# ## NumPy's eingebaute Lösung
#
# NumPy hat eine Funktion dafür: `np.linalg.lstsq()`

# %% [markdown]
#
# Verwende NumPys Least-Squares-Solver

# %%
theta_numpy, residuals, rank, s = np.linalg.lstsq(
    X_with_intercept, y_simple, rcond=None
)

# %%
print("NumPy's solution:")
print(f"Intercept: {theta_numpy[0]:.3f}")
print(f"Coefficient: {theta_numpy[1]:.3f}")

# %% [markdown]
#
# Zusätzliche Informationen

# %%
print(f"\nResiduals (sum of squared errors): {residuals[0]:.3f}")
print(f"Rank of matrix: {rank}")


# %% [markdown]
#
# ## Implementierung der fit() Methode

# %%
class TrainableLinearRegression:
    """Linear regression model that can learn from data."""

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Learn coefficients from training data.

        Parameters:
        - X: 2D array of features (n_samples, n_features)
        - y: 1D array of targets (n_samples,)

        Returns:
        - self (for method chaining)
        """
        X = np.array(X)
        y = np.array(y)

        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(n_samples), X])

        # Solve using normal equation
        theta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)

        # Extract coefficients and intercept
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions!")

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.coef_ + self.intercept_


# %% [markdown]
#
# ## Unser Modell testen
#
# Erstelle und trainiere unser Modell

# %%
our_model = TrainableLinearRegression()

# %%
our_model.fit(X_simple, y_simple)

# %%
print("Learned parameters:")
print(f"Coefficient: {our_model.coef_[0]:.3f}")
print(f"Intercept: {our_model.intercept_:.3f}")

# %% [markdown]
#
# Berechne Vorhersagen

# %%
predictions = our_model.predict(X_simple)

# %%
print("\nPredictions:", predictions)
print("True values:", y_simple)

# %% [markdown]
#
# ### Berechne den Fehler

# %%
mse = np.mean((y_simple - predictions) ** 2)

# %%
print(f"\nMSE: {mse:.6f}")

# %% [markdown]
#
# ## Visualisierung der gelernten Funktion
#
# - Plotte Trainingsdaten
# - Plotte gelernte Linie

# %%
x_line = np.linspace(0, 6, 100).reshape(-1, 1)
y_line = our_model.predict(x_line)

# %%
coef = our_model.coef_[0]
intercept = our_model.intercept_

# %%
model_label = f"y = {coef:.2f}x + {intercept:.2f}"

# %%
plt.figure(figsize=(8, 5))
plt.scatter(X_simple, y_simple, s=100, alpha=0.7, label="Training data")
plt.plot(x_line, y_line, "r-", linewidth=2, label=model_label)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Learned Linear Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
#
# ## Multivariate Regression
#
# Komplexeres Beispiel mit mehreren Features

# %%
np.random.seed(42)
n_samples = 100
n_features = 3

# %%
X_multi = np.random.randn(n_samples, n_features)

# %%
true_coef = np.array([2.5, -1.5, 3.0])
true_intercept = 5.0

# %%
noise = np.random.randn(n_samples) * 0.25

# %%
y_multi = X_multi @ true_coef + true_intercept + noise

# %% [markdown]
#
# Trainiere unser Modell

# %%
multi_model = TrainableLinearRegression()
multi_model.fit(X_multi, y_multi)

# %%
print(f"Dataset: {n_samples} samples, {n_features} features")
print("True parameters:")
print(f"  True coefficients: {true_coef}")
print(f"  True intercept: {true_intercept}")
print()
print("Learned parameters:")
print(f"  Coefficients: {multi_model.coef_}")
print(f"  Intercept: {multi_model.intercept_:.3f}")

# %%
print("\nCompare with true values:")
for i, (learned, true) in enumerate(zip(multi_model.coef_, true_coef)):
    error = abs(learned - true)
    print(f"Coef {i}: Learned={learned:.3f}, True={true:.3f}, Error={error:.3f}")


# %% [markdown]
#
# ## Grenzen der Normalgleichung
#
# **Vorteile:**
# - Exakte Lösung in einem Schritt
# - Keine Hyperparameter
# - Schnell für kleine Datensätze
#
# **Nachteile:**
# - Langsam für große Datensätze (Matrix-Inversion ist O(n³))
# - Funktioniert nicht wenn X^T×X nicht invertierbar ist
# - Speicherintensiv für viele Features

# %% [markdown]
#
# ## Alternative: Gradient Descent
#
# Für große Datensätze verwendet man iterative Methoden:
# - Starte mit zufälligen Koeffizienten
# - Verbessere schrittweise
# - Folge dem Gradienten bergab
# - Basis für neuronale Netze!

# %% [markdown]
#
# Visualisiere Gradient-Descent-Konzept
#
# - Fehleroberfläche (1D zur Visualisierung)
# - Schritte zum Minimum

# %%
from plot_utils import plot_gradient_descent_concept

# %%
plot_gradient_descent_concept()

# %% [markdown]
#
# ## Zusammenfassung
#
# - Modelle lernen durch Minimierung der Fehler
# - Die Normalgleichung findet optimale Koeffizienten direkt
# - $θ = (X^T × X)^{-1} × X^T × y$
# - NumPy's `lstsq()` macht die Berechnung einfach
# - Grundlage für komplexere Lernalgorithmen!

# %% [markdown]
#
# ## Was haben wir gelernt?
#
# 1. **Vektoren und Matrizen** - Die Sprache des ML
# 2. **Skalarprodukt** - Kernoperation für Vorhersagen
# 3. **Matrix-Multiplikation** - Batch-Vorhersagen
# 4. **Modellarchitektur** - Koeffizienten + Intercept
# 5. **Lernen aus Daten** - Normalgleichung
#
# **Bereit für neuronale Netze!**

# %% [markdown]
#
# ## Workshop: Vollständiges Modell
#
# Wohnungsdaten für den  Workshop:

# %%
from sklearn.datasets import make_regression

# %%
X_housing, y_housing = make_regression(
    n_samples=50, n_features=4, noise=10, random_state=42
)

# %%
feature_names = ["Size", "Bedrooms", "Location", "Age"]

# %%
print(f"Housing dataset: {X_housing.shape[0]} houses, {X_housing.shape[1]} features")

# %% [markdown]
#
# ### Aufgabe 1: Modell trainieren
#
# Trainieren Sie ein Modell auf den Wohnungsdaten:

# %% [markdown]
#
# Trainiere Modell

# %%
housing_model = TrainableLinearRegression()
housing_model.fit(X_housing, y_housing)

# %%
print("Trained Housing Model:")
print("-" * 40)
for name, coef in zip(feature_names, housing_model.coef_):
    print(f"{name:10}: {coef:+8.2f}")
print(f"{'Intercept':10}: {housing_model.intercept_:+8.2f}")

# %% [markdown]
#
# ### Aufgabe 2: Modellqualität bewerten
#
# Bewerten Sie die Qualität des Modells.

# %% [markdown]
#
# Mache Vorhersagen

# %%
predictions = housing_model.predict(X_housing)

# %% [markdown]
#
# Berechne Metriken

# %%
errors = y_housing - predictions
mse = np.mean(errors**2)
mae = np.mean(np.abs(errors))
rmse = np.sqrt(mse)

# %%
print("Model Performance:")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

# %% [markdown]
#
# Visualisiere Vorhersagen vs. tatsächliche Werte

# %%
plt.figure(figsize=(8, 6))
plt.scatter(y_housing, predictions, alpha=0.6)
plt.plot(
    [y_housing.min(), y_housing.max()],
    [y_housing.min(), y_housing.max()],
    "r--",
    linewidth=2,
)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Predictions vs True Values")
plt.grid(True, alpha=0.3)
plt.show()
