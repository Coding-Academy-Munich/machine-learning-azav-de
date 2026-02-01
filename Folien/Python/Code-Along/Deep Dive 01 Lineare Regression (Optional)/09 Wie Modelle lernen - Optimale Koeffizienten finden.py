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

# %% [markdown]
#
# ## Ein einfaches Beispiel

# %%
import install_packages

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
from linear_regression_plots import plot_data

# %%
plot_data(X_simple, y_simple)

# %% [markdown]
# <br><br><br><br><br><br>

# %% [markdown]
#
# ## Was sind "gute" Koeffizienten?
#
# Wir wollen eine Linie, die:
# - Möglichst nah an allen Punkten ist
# - Die Fehler minimiert
# - Gut generalisiert

# %%
from linear_regression_plots import plot_coefficients_effect, plot_coefficients_interactive

# %%
plot_coefficients_interactive(X_simple, y_simple)

# %% [markdown]
# <br><br><br><br><br><br>

# %%
coefficients_to_try = [1.5, 2.0, 2.5]
intercepts_to_try = [1.5, 1.0, 0.5]

# %%
plot_coefficients_effect(X_simple, y_simple, coefficients_to_try, intercepts_to_try)

# %% [markdown]
# <br>

# %% [markdown]
#
# ## Die Kosten von Fehlern
#
# **Mean Squared Error (MSE)**:
# - Misst durchschnittliche quadratische Abweichung
# - Bestraft große Fehler stärker
# - MSE = $\frac{1}{n} \times \sum(y_{\textrm{true}} - y_{\textrm{pred}})^2$

# %% [markdown]
#
# ## Visualisierung der Fehleroberfläche
#
# - 3D-Fehleroberfläche
# - Konturplot


# %%
from linear_regression_plots import plot_error_surface_3d, plot_error_contour

# %%
best_coef, best_intercept, min_mse = plot_error_surface_3d(X_simple, y_simple)

# %%
print(f"Best coefficient: {best_coef:.3f}")
print(f"Best intercept: {best_intercept:.3f}")
print(f"Minimum MSE: {min_mse:.3f}")

# %%
best_coef, best_intercept, min_mse = plot_error_surface_3d(
    X_simple, y_simple, clip_percentile=75
)

# %%
best_coef, best_intercept, min_mse = plot_error_surface_3d(
    X_simple, y_simple, use_log_scale=True, log_epsilon=0.5
)

# %% [markdown]
# <br><br><br><br><br><br>

# %%
plot_error_contour(X_simple, y_simple)

# %% [markdown]
# <br><br><br><br><br><br>


# %% [markdown]
#
# ### Daten mit Abweichungen (Fehler/Rauschen/...)
#
# - Fügen wir zufälliges Rauschen hinzu
# - Simuliert reale Daten

# %%
np.random.seed(42)
noise = np.random.randn(len(y_simple))

# %%
y_noisy = y_simple + noise

# %%
plot_data(X_simple, y_noisy)

# %% [markdown]
# <br><br><br><br><br><br>

# %%
plot_coefficients_interactive(X_simple, y_noisy)

# %% [markdown]
# <br><br><br><br><br><br>

# %%
best_coef_noisy, best_intercept_noisy, min_mse_noisy = plot_error_surface_3d(
    X_simple, y_noisy
)

# %%
print(f"Best coefficient (noisy): {best_coef_noisy:.3f}")
print(f"Best intercept (noisy): {best_intercept_noisy:.3f}")
print(f"Minimum MSE (noisy): {min_mse_noisy:.3f}")

# %% [markdown]
# <br><br><br><br><br><br>

# %%
plot_error_contour(X_simple, y_noisy)

# %% [markdown]
# <br><br><br><br><br><br>

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
# ## Herleitung der Normalgleichung (Teil 1)
#
# **Ausgangspunkt: Die Kostenfunktion**
#
# Notation:
# - $m$ = Anzahl der Trainingsbeispiele
# - $n$ = Anzahl der Features
# - $x^{(i)}$ = Feature-Vektor für Trainingsbeispiel $i$ (Dimension $n$)
# - $y^{(i)}$ = Wahrer Zielwert für Beispiel $i$
# - $θ$ = Parameter-Vektor (Koeffizienten, Dimension $n$)
# - $h_θ(x) = θ^T x$ = Vorhersage (Hypothese)

# %% [markdown]
#
# ## Herleitung der Normalgleichung (Teil 1, Forts.)
#
# **Die Kostenfunktion**
#
# Wir wollen den Mean Squared Error minimieren:
#
# $$J(θ) = \frac{1}{2m} \sum_{i=1}^{m} (h_θ(x^{(i)}) - y^{(i)})^2$$
#
# Dies misst den durchschnittlichen quadratischen Fehler über alle Trainingsbeispiele.

# %% [markdown]
#
# ## Herleitung der Normalgleichung (Teil 2)
#
# **Matrixform der Kostenfunktion**
#
# Wir können alle Trainingsbeispiele in Matrizen zusammenfassen:
#
# - $X$ = Matrix aller Trainingsbeispiele $(m × n)$
#   - Jede Zeile ist ein $x^{(i)}$
# - $θ$ = Vektor der Parameter $(n × 1)$
# - $y$ = Vektor aller Zielwerte $(m × 1)$
#
# In Matrixschreibweise:
#
# $$J(θ) = \frac{1}{2m} (Xθ - y)^T(Xθ - y)$$

# %% [markdown]
#
# ## Herleitung der Normalgleichung (Teil 3)
#
# **Ausmultiplizieren**
#
# Wir entwickeln die Kostenfunktion:
#
# $$
# \begin{align*}
# J(θ) & = \frac{1}{2m} (Xθ - y)^T(Xθ - y) \\
# & = \frac{1}{2m} ((Xθ)^T - y^T)(Xθ - y) \\
# & = \frac{1}{2m} [(Xθ)^T(Xθ) - (Xθ)^T y - y^T(Xθ) + y^T y]
# \end{align*}
# $$
#
# Da $(Xθ)^T y$ und $y^T(Xθ)$ beide Skalare sind und gleich:
#
# $$= \frac{1}{2m} [θ^T X^T X θ - 2(Xθ)^T y + y^T y]$$

# %% [markdown]
#
# ## Herleitung der Normalgleichung (Teil 4)
#
# **Ableiten nach θ**
#
# Um das Minimum zu finden, setzen wir die Ableitung auf 0:
#
# $$\frac{\partial J(θ)}{\partial θ} = 0$$
#
# Regeln der Matrixableitung (für symmetrisches $A$ und Vektor $a$):
# - $\frac{\partial}{\partial θ}(θ^T A θ) = 2Aθ$
# - $\frac{\partial}{\partial θ}(a^T θ) = a$
# - $\frac{\partial}{\partial θ}(\text{const}) = 0$

# %% [markdown]
#
# ## Herleitung der Normalgleichung (Teil 4, Forts.)
#
# **Anwendung der Ableitungsregeln**
#
# Aus $J(θ) = \frac{1}{2m} [θ^T X^T X θ - 2(Xθ)^T y + y^T y]$ erhalten wir:
#
# $$\frac{\partial J(θ)}{\partial θ} = \frac{1}{2m}[2X^T X θ - 2X^T y] = 0$$

# %% [markdown]
#
# ## Herleitung der Normalgleichung (Teil 5)
#
# **Auflösen nach θ**
#
# $$\frac{1}{2m}[2X^T X θ - 2X^T y] = 0$$
#
# Vereinfachen (Konstanten kürzen sich):
#
# $$X^T X θ - X^T y = 0$$
#
# $$X^T X θ = X^T y$$
#
# Multipliziere beide Seiten mit $(X^T X)^{-1}$:
#
# $$θ = (X^T X)^{-1} X^T y$$
#
# **Das ist die Normalgleichung!**

# %% [markdown]
#
# ## Intuition hinter der Normalgleichung
#
# **Geometrische Interpretation:**
#
# - $Xθ$ ist unsere Vorhersage (liegt im Spaltenraum von $X$)
# - $y$ ist der wahre Wert (kann außerhalb des Spaltenraums liegen)
# - Wir suchen das $θ$, sodass $Xθ$ am nächsten zu $y$ ist
# - Das ist die orthogonale Projektion von $y$ auf den Spaltenraum von $X$
#
# **Bedingung für Projektion:**
# - Der Fehler $(y - Xθ)$ muss senkrecht zum Spaltenraum von $X$ stehen
# - Mathematisch: $X^T(y - Xθ) = 0$
# - Dies führt direkt zur Normalgleichung!

# %% [markdown]
#
# ## Geometrische Interpretation: Ein einfaches Beispiel
#
# Betrachten wir ein minimales Beispiel:
#
# - 3 Datenpunkte, 1 Feature
# - Intercept-Spalte hinzugefügt
# - Visualisierung der Projektion und des Fehlers

# %%
X_geo = np.array([[1, -2], [1, 2], [1, 2]])
y_geo = np.array([-0.5, -1.0, 3.5])

# %% [markdown]
#
# Spaltenraum von X (2D-Ebene in 3D-Raum):
#
# - Die Spalten von X spannen den Spaltenraum auf

# %%
col1 = X_geo[:, 0]
col2 = X_geo[:, 1]

# %%
print("Column 1 (intercept):", col1)
print("Column 2 (feature):", col2)
print("Target vector y:", y_geo)

# %% [markdown]
#
# ## Visualisierung der Projektion
#
# - y liegt außerhalb des Spaltenraums
# - Xθ ist die Projektion von y auf den Spaltenraum
# - Der Fehler (y - Xθ) steht senkrecht auf dem Spaltenraum

# %%
from linear_regression_plots import plot_geometric_projection

# %%
theta_geo, y_pred_geo, error_geo = plot_geometric_projection(X_geo, y_geo)

# %%
print(f"Optimal θ: {theta_geo}")
print(f"Prediction Xθ: {y_pred_geo}")
print(f"Error (y - Xθ): {error_geo}")

# %% [markdown]
# <br><br><br><br><br><br>

# %% [markdown]
#
# ## Überprüfung der Orthogonalität
#
# Der Fehler sollte senkrecht auf beiden Spalten von X stehen:

# %%
# Check that error is perpendicular to column space
# Dot product should be (close to) zero
dot_col1 = np.dot(col1, error_geo)
dot_col2 = np.dot(col2, error_geo)

# %%

# %% [markdown]
#
# ## Was zeigt diese Visualisierung?
#
# - **Blaue und grüne Pfeile**: Die beiden Spalten von X spannen eine Ebene auf
# - **Roter Pfeil**: Der Zielvektor y (liegt außerhalb der Ebene)
# - **Lila Pfeil**: Die Vorhersage Xθ (liegt in der Ebene - optimale Projektion)
# - **Oranger Pfeil**: Der Fehler (steht senkrecht auf der Ebene!)
#
# Die Normalgleichung findet automatisch die Koeffizienten θ, sodass:
# - Xθ die orthogonale Projektion von y ist
# - Der Fehler minimal wird
# - Der Fehler senkrecht auf dem Spaltenraum steht

# %% [markdown]
#
# ## Wann existiert die Lösung?
#
# **Voraussetzungen für $(X^T X)^{-1}$:**
#
# - Die Matrix $X^T X$ muss invertierbar sein.
# - Das bedeutet nicht, dass $X$ selbst invertierbar sein muss (kann rechteckig sein).
#
# **In der Praxis:** NumPy's `lstsq()` verwendet SVD (Singular Value Decomposition)
# und kann auch mit nicht-invertierbaren Matrizen umgehen!

# %% [markdown]
#
# ## NumPy's `lstsq`: Die praktische Lösung
#
# **Warum nicht direkt die Normalgleichung implementieren?**
#
# Die Normalgleichung $θ = (X^T X)^{-1} X^T y$ hat in der Praxis Probleme:
# - Matrix-Inversion ist numerisch instabil
# - $(X^T X)$ kann singulär oder fast singulär sein
# - Ineffizient bei großen Matrizen
#
# **Bessere Lösung:** `np.linalg.lstsq()`

# %% [markdown]
#
# ### Was macht `lstsq`?
#
# **Least Squares** = Kleinste Quadrate
#
# - Löst die Normalgleichung **ohne explizite Inversion**
# - Verwendet SVD (Singular Value Decomposition)
# - Findet die Lösung, die den quadratischen Fehler minimiert
# - Robust und numerisch stabil
# - Funktioniert auch wenn $X^T X$ nicht invertierbar ist!

# %% [markdown]
#
# ### Wie `lstsq` die Normalgleichung löst
#
# **Erinnerung - Normalgleichung:**
# $$X^T X θ = X^T y$$
#
# **`lstsq` macht folgendes:**
# 1. Zerlegt $X$ mittels SVD: $X = U Σ V^T$
# 2. Berechnet $θ$ ohne $(X^T X)^{-1}$ zu bilden
# 3. Minimiert $\|Xθ - y\|^2$ (genau wie die Normalgleichung!)
#
# **Resultat:** Dieselbe Lösung wie die Normalgleichung, aber numerisch stabiler!

# %% [markdown]
#
# ### Beispiel: Vergleich der Methoden
#
# Lösen wir ein einfaches System auf beide Arten:
#
# - 1 Unbekannte + Intercept, 3 Gleichungen
# - Exakte Lösung existiert

# %%
X_compare = np.array([[1, 1], [1, 2], [1, 3]])
y_compare = np.array([2, 4, 6])

# %%
X_compare

# %%
y_compare

# %% [markdown]
#
# **Methode 1: Normalgleichung (direkte Inversion)**
#
# $$ θ = (X^T X)^{-1} X^T y $$

# %%

# %%

# %%

# %%

# %% [markdown]
#
# **Methode 2: `lstsq` (SVD-basiert)**
#
# - Gibt direkt die Lösung zurück

# %%

# %%

# %% [markdown]
#
# **Vergleich der Ergebnisse:**

# %%
print("Normal equation solution:", theta_normal)
print("lstsq solution:", theta_lstsq)
print("Are they equal?", np.allclose(theta_normal, theta_lstsq))


# %% [markdown]
#
# ## Systeme ohne exakte Lösung
#
# - In ML haben wir oft Systeme, für die keine exakte Lösung existiert
#   - Rauschen in den Daten
#   - Zu wenige Features
# - Die Normalengleichung und `lstsq` finden die **beste Approximation**!

# %% [markdown]
#
# ### Beispiel: Verrauschte Daten

# %%

# %%

# %%

# %% [markdown]
#
# Die Normalengleichung findet die beste Gerade durch die verrauschten Punkte:

# %%

# %%

# %% [markdown]
#
# `lstsq` findet auch die beste Gerade durch die verrauschten Punkte:

# %%

# %%

# %% [markdown]
#
# Visualisierung:

# %%
def plot_data(X, y, theta):
    y_pred = X @ theta

    plt.figure(figsize=(8, 5))
    plt.scatter([1, 2, 3, 4, 5], y, color="blue", s=100, label="Noisy data", zorder=3)
    plt.plot(
        [1, 2, 3, 4, 5], y_pred, "r-", linewidth=2, label="Best fit (lstsq)", zorder=2
    )

    # Show errors
    for i in range(len(y)):
        plt.plot([i + 1, i + 1], [y[i], y_pred[i]], "g--", alpha=0.5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Least Squares Fit: y = {theta[0]:.2f} + {theta[1]:.2f}x")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# %%

# %% [markdown]
#
# ## Zusätzliche Informationen von `lstsq`
#
# `lstsq` gibt nicht nur die Lösung zurück, sondern auch:

# %% [markdown]
#
# 1. **Lösung (theta):** Die optimalen Parameter
# 2. **Residuals:** Summe der quadrierten Fehler (für überbestimmte Systeme)
# 3. **Rank:** Rang der Matrix (Anzahl linear unabhängiger Spalten)
# 4. **Singular values:** Singuläre Werte aus der SVD
#
# Diese Informationen helfen, die Qualität der Lösung zu bewerten!

# %% [markdown]
#
# ### Interpretation der Ausgaben

# %%

# %%

# %% [markdown]
#
# ## Warum `lstsq` in Machine Learning?
#
# **Die Verbindung zur linearen Regression:**
#
# - Unsere Normalgleichung: $θ = (X^T X)^{-1} X^T y$
#   - Löst $X θ = y$ wenn möglich
#   - Minimiert MSE $\|X θ - y\|^2$
# - `lstsq` löst $X θ = y$ wenn möglich
#   - Minimiert MSE $\|X θ - y\|^2$
#   - Funktioniert auch wenn $X^T X$ nicht invertierbar ist
# - Perfekt für lineare Regression mit vielen Daten!

# %% [markdown]
#
# **Deshalb verwenden wir `lstsq` in unserer `fit()` Methode:**
#
# ```python
# # Statt:
# theta = np.linalg.inv(X.T @ X) @ X.T @ y  # Instabil!
#
# # Verwenden wir:
# theta, _, _, _ = np.linalg.lstsq(X, y)  # Robust!
# ```
#
# **Gleiche mathematische Lösung, aber besser implementiert!**

# %% [markdown]
#
# ## Zusammenfassung: Normalgleichung und `lstsq`
#
# **Mathematik:**
# - Normalgleichung: $θ = (X^T X)^{-1} X^T y$
# - Minimiert Mean Squared Error
# - Findet optimale Lösung analytisch
#
# **Praxis:**
# - `np.linalg.lstsq()` für numerische Stabilität
# - Löst dasselbe Problem ohne direkte Inversion
# - Funktioniert auch bei schwierigen Fällen
# - **Das verwenden wir in der Implementation!**

# %% [markdown]
#
# ## Schritt 1: Daten vorbereiten
#
# Für den Intercept fügen wir eine Spalte mit Einsen hinzu:

# %%

# %%

# %%

# %% [markdown]
#
# ## Schritt 2: Normalgleichung anwenden

# %% [markdown]
#
# Normalgleichung: $θ = (X^T X)^{-1} X^T y$

# %% [markdown]
#
# Verwende NumPys Least-Squares-Solver

# %%

# %%
print(f"Intercept: {theta[0]:.3f}")
print(f"Coefficient: {theta[1]:.3f}")

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

# %% [markdown]
#
# ## Unser Modell testen
#
# Erstelle und trainiere unser Modell

# %%

# %%

# %%

# %% [markdown]
#
# Berechne Vorhersagen

# %%

# %%

# %% [markdown]
#
# ### Berechne den Fehler

# %%

# %%

# %% [markdown]
#
# ## Visualisierung der gelernten Funktion
#

# %%
def plot_learned_function(X, y, model):
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="blue", s=100, label="Data points", zorder=3)
    plt.plot(X, y_pred, "r-", linewidth=2, label="Learned function", zorder=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Learned Linear Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# %%


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
from linear_regression_plots import plot_gradient_descent_concept

# %%
plot_gradient_descent_concept()

# %% [markdown]
# <br>

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

# %%

# %% [markdown]
#
# ### Aufgabe 2: Modellqualität bewerten
#
# Bewerten Sie die Qualität des Modells.

# %%

# %%
