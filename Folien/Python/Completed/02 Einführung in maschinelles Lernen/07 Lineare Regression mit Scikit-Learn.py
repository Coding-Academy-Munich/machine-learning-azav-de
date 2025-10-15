# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Lineare Regression mit Scikit-Learn</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Beispiel: Lineare Regression mit Scikit-Learn
#
# Wir verwenden das gleiche Beispiel wie vorher:

# %%
data_x = [0.5, 1.4, 2.1, 3.5, 4.2, 5.1, 6.3, 7.6, 8.2, 9.0]
data_y = [2.1, 3.5, 5.8, 7.1, 9.3, 11.5, 13.2, 16.0, 17.1, 19.2]

# %% [markdown]
#
# ## Train-Test Split
#
# ### Warum brauchen wir einen Train-Test Split?
#
# - Nach dem Training muss das Modell bewertet werden
# - **Die Bewertung muss auf anderen Daten basieren als das Training**
# - Andernfalls könnte das Modell die Trainingsdaten "auswendig lernen" (Overfitting)
# - Uns interessiert, wie gut das Modell auf **neuen, ungesehenen** Daten funktioniert

# %% [markdown]
#
# ## Warum die Daten randomisieren?
#
# - Daten sind oft in einer bestimmten Reihenfolge gesammelt
#   - Zeitliche Reihenfolge
#   - Sortiert nach bestimmten Eigenschaften
#   - Gruppiert nach Kategorien
# - Wenn wir die Daten nicht randomisieren, könnten Train- und Testdaten
#   systematisch unterschiedlich sein
# - Beispiel: Erste 80% = Sommer, letzte 20% = Winter
# - Randomisierung stellt sicher, dass beide Datensätze repräsentativ sind

# %% [markdown]
#
# ## Wie viel Prozent für Training und Test?
#
# **Typische Aufteilungen:**
# - 80/20 (80% Training, 20% Test): Standard für mittelgroße Datensätze
# - 70/30: Wenn mehr Testdaten gewünscht sind
# - 90/10: Wenn wenig Daten vorhanden sind
#
# **Faktoren bei der Entscheidung:**
# - Größe des Datensatzes: Je weniger Daten, desto mehr für Training
# - Komplexität des Modells: Komplexe Modelle brauchen mehr Trainingsdaten
# - Variabilität der Daten: Hohe Variabilität erfordert mehr Testdaten

# %% [markdown]
#
# ## Train-Test Split mit `train_test_split()`
#
# Scikit-Learn bietet die Funktion `train_test_split()` an:
#
# - Teilt die Daten automatisch auf
# - Randomisiert die Daten standardmäßig
# - Kann mit `random_state` für reproduzierbare Ergebnisse fixiert werden
# - Parameter `test_size`: Anteil der Testdaten (0.0 bis 1.0)

# %%
from sklearn.model_selection import train_test_split

# %% [markdown]
#
# ## Vorbereitung der Daten
#
# - `train_test_split()` erwartet X als 2D-Array (Liste von Listen)
# - Jede Zeile ist ein Datenpunkt, jede Spalte ein Feature

# %%
# Convert to 2D format (required by scikit-learn)
X = [[x] for x in data_x]
y = data_y

# %%
X[:5]

# %% [markdown]
#
# ## Durchführung des Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# %% [markdown]
#
# ## Was bedeutet `random_state=42`?
#
# - Die Randomisierung ist eigentlich pseudozufällig
# - `random_state` setzt den "Seed" für den Zufallsgenerator
# - Mit dem gleichen `random_state` bekommen wir immer die gleiche Aufteilung
# - Wichtig für:
#   - Reproduzierbarkeit von Ergebnissen
#   - Debugging
#   - Vergleich verschiedener Modelle

# %%
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Training percentage: {len(X_train) / len(X) * 100:.1f}%")

# %%
print("Training data:")
print("X_train:", X_train)
print("y_train:", y_train)

# %%
print("Test data:")
print("X_test:", X_test)
print("y_test:", y_test)

# %% [markdown]
#
# ## Training des Modells mit Scikit-Learn

# %%
from sklearn.linear_model import LinearRegression

# %%
model = LinearRegression()

# %% [markdown]
#
# ## Vor dem Training
#
# Das Modell ist noch nicht trainiert:

# %%
try:
    print(f"Coefficient: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
except AttributeError as e:
    print(f"Error: {e}")
    print("Model is not fitted yet.")

# %% [markdown]
#
# ## Training mit `fit()`

# %%
model.fit(X_train, y_train)

# %% [markdown]
#
# ## Nach dem Training
#
# Jetzt sind die Parameter gesetzt:

# %%
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# %% [markdown]
#
# ## Interpretation
#
# - Das Modell hat die Gleichung: y = 2x + 0.9 (ungefähr)
# - Für jede Erhöhung von x um 1 steigt y um etwa 2
# - Der y-Achsenabschnitt ist etwa 0.9

# %% [markdown]
#
# ## Evaluation mit Scikit-Learn
#
# Scikit-Learn bietet viele Metriken zur Modellbewertung:
#
# ### Für Regression:
# - **Mean Absolute Error (MAE)**: Durchschnittlicher absoluter Fehler
# - **Mean Squared Error (MSE)**: Durchschnittlicher quadratischer Fehler
# - **Root Mean Squared Error (RMSE)**: Wurzel des MSE
# - **R² Score**: Bestimmtheitsmaß (0 bis 1, höher ist besser)

# %% [markdown]
#
# ## Mean Absolute Error (MAE)
#
# - Durchschnitt der absoluten Differenzen zwischen Vorhersage und tatsächlichem Wert
# - **Einheit**: Gleiche Einheit wie die Zielvariable
# - **Interpretation**: Durchschnittliche Abweichung der Vorhersagen
# - **Vorteil**: Einfach zu interpretieren, nicht anfällig für Ausreißer
# - **Formel**: $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

# %% [markdown]
#
# ## Mean Squared Error (MSE)
#
# - Durchschnitt der quadrierten Differenzen
# - **Einheit**: Quadrat der Einheit der Zielvariable
# - **Interpretation**: Bestraft große Fehler stärker
# - **Vorteil**: Mathematisch besser zu handhaben (differenzierbar)
# - **Nachteil**: Schwerer zu interpretieren durch Quadrierung
# - **Formel**: $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

# %% [markdown]
#
# ## Root Mean Squared Error (RMSE)
#
# - Wurzel des MSE
# - **Einheit**: Gleiche Einheit wie die Zielvariable
# - **Interpretation**: Ähnlich wie MAE, aber bestraft große Fehler stärker
# - **Vorteil**: Kombiniert Vorteile von MAE (Einheit) und MSE (Gewichtung)
# - **Formel**: $\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

# %% [markdown]
#
# ## R² Score (Bestimmtheitsmaß)
#
# - Misst, wie gut das Modell die Varianz in den Daten erklärt
# - **Bereich**: -∞ bis 1 (normalerweise 0 bis 1)
#   - 1: Perfekte Vorhersage
#   - 0: Modell ist nicht besser als der Durchschnitt
#   - < 0: Modell ist schlechter als der Durchschnitt
# - **Vorteil**: Unabhängig von der Einheit, leicht zu interpretieren
# - **Formel**: $R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$

# %% [markdown]
#
# ## Vorhersagen machen

# %%
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# %%
print("Training predictions:", train_predictions[:5])
print("Actual training values:", y_train[:5])

# %% [markdown]
#
# ## Evaluation auf Trainingsdaten

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# %%
train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = root_mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

# %%
print(f"Training Metrics:")
print(f"  MAE:  {train_mae:.4f}")
print(f"  MSE:  {train_mse:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  R²:   {train_r2:.4f}")

# %% [markdown]
#
# ## Evaluation auf Testdaten
#
# **Wichtig**: Die Performance auf den Testdaten ist die aussagekräftigere
# Metrik!

# %%
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = root_mean_squared_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

# %%
print(f"Test Metrics:")
print(f"  MAE:  {test_mae:.4f}")
print(f"  MSE:  {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  R²:   {test_r2:.4f}")

# %% [markdown]
#
# ## Vergleich: Training vs. Test
#
# Ein gutes Modell sollte ähnliche Performance auf beiden Datensätzen zeigen:

# %%
print("Comparison:")
print(f"{'Metric':<10} {'Training':<12} {'Test':<12} {'Difference':<12}")
print("-" * 50)
print(f"{'MAE':<10} {train_mae:<12.4f} {test_mae:<12.4f} {abs(train_mae - test_mae):<12.4f}")
print(f"{'MSE':<10} {train_mse:<12.4f} {test_mse:<12.4f} {abs(train_mse - test_mse):<12.4f}")
print(f"{'RMSE':<10} {train_rmse:<12.4f} {test_rmse:<12.4f} {abs(train_rmse - test_rmse):<12.4f}")
print(f"{'R²':<10} {train_r2:<12.4f} {test_r2:<12.4f} {abs(train_r2 - test_r2):<12.4f}")

# %% [markdown]
#
# ## Interpretation der Ergebnisse
#
# - **R² nahe 1**: Das Modell erklärt die Daten sehr gut
# - **Kleine MAE/RMSE**: Die Vorhersagen sind im Durchschnitt nah an den wahren Werten
# - **Training ≈ Test**: Kein Overfitting, das Modell generalisiert gut
# - **Training ≫ Test**: Mögliches Overfitting, Modell ist zu komplex

# %% [markdown]
#
# ## Visualisierung der Ergebnisse

# %%
X_pred = [x[0] for x in X_train + X_test]
y_pred = model.predict(X_train + X_test)

# %%
from matplotlib import pyplot as plt

# %%
def plot_data_and_model():
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
    plt.scatter(X_test, y_test, alpha=0.6, label='Test data')
    plt.plot(X_pred, y_pred, 'r-', label='Model predictions', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Linear Regression: Model Fit')
    plt.show()

# %%
plot_data_and_model()

# %% [markdown]
#
# ## Residuen-Plot
#
# - Zeigt die Fehler (Residuen) für jede Vorhersage
# - Ideale Residuen sind zufällig um 0 verteilt
# - Muster in den Residuen deuten auf Probleme hin

# %%
train_residuals = [y - pred for y, pred in zip(y_train, train_predictions)]
test_residuals = [y - pred for y, pred in zip(y_test, test_predictions)]

# %%
def plot_residuals():
    plt.figure(figsize=(10, 6))
    plt.scatter(train_predictions, train_residuals, alpha=0.6, label='Training')
    plt.scatter(test_predictions, test_residuals, alpha=0.6, label='Test')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend()
    plt.title('Residual Plot')
    plt.show()

# %%
plot_residuals()

# %% [markdown]
#
# ## Weitere nützliche Scikit-Learn Tools
#
# ### Cross-Validation
# - Teilt Daten in mehrere Folds auf
# - Trainiert und testet Modell mehrfach
# - Robustere Schätzung der Modell-Performance
#
# ```python
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model, X, y, cv=5)
# ```

# %% [markdown]
#
# ### Grid Search
# - Systematische Suche nach besten Hyperparametern
# - Testet verschiedene Kombinationen von Parametern
#
# ```python
# from sklearn.model_selection import GridSearchCV
# param_grid = {'fit_intercept': [True, False]}
# grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5)
# grid_search.fit(X, y)
# ```

# %% [markdown]
#
# ### Pipelines
# - Verketten mehrere Verarbeitungsschritte
# - Preprocessing + Training in einem Schritt
#
# ```python
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
#
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('model', LinearRegression())
# ])
# pipeline.fit(X_train, y_train)
# ```

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Scikit-Learn** ist eine mächtige und einfach zu benutzende ML-Bibliothek
# - **Train-Test Split** ist essentiell zur Bewertung von Modellen
#   - Randomisierung verhindert systematische Verzerrungen
#   - Typische Aufteilung: 80/20 oder 70/30
# - **Evaluation** sollte immer auf den Testdaten erfolgen
#   - MAE, MSE, RMSE für absolute Fehler
#   - R² für relative Modellgüte
# - Scikit-Learn bietet viele weitere Tools (Cross-Validation, Grid Search, Pipelines)
