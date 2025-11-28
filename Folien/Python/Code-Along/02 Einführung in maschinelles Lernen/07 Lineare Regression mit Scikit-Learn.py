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

# %% [markdown]
#
# ## Vorbereitung der Daten
#
# - `train_test_split()` erwartet X als 2D-Array (Liste von Listen)
# - Jede Zeile ist ein Datenpunkt, jede Spalte ein Feature

# %%

# %%

# %% [markdown]
#
# ## Durchführung des Train-Test Split

# %%

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

# %%

# %% [markdown]
#
# ## Vor dem Training
#
# Das Modell ist noch nicht trainiert:

# %%

# %% [markdown]
#
# ## Training mit `fit()`

# %%

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

# %%

# %% [markdown]
#
# ## Evaluation auf Trainingsdaten

# %%

# %%

# %%

# %% [markdown]
#
# ## Evaluation auf Testdaten
#
# **Wichtig**: Die Performance auf den Testdaten ist die aussagekräftigere
# Metrik!

# %%

# %%

# %% [markdown]
#
# ## Vergleich: Training vs. Test
#
# Ein gutes Modell sollte ähnliche Performance auf beiden Datensätzen zeigen:

# %%

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
    plt.scatter(X_train, y_train, alpha=0.6, label="Training data")
    plt.scatter(X_test, y_test, alpha=0.6, label="Test data")
    plt.plot(X_pred, y_pred, "r-", label="Model predictions", linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Linear Regression: Model Fit")
    plt.show()


# %%

# %% [markdown]
#
# ## Evaluierung mit `.score()`
#
# - Die Methode `.score()` bietet eine schnelle Möglichkeit, die Leistung des
#   Modells zu bewerten.
# - Hier gibt sie das Bestimmtheitsmaß R² der Vorhersage zurück.

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Residuen-Plot
#
# - Zeigt die Fehler (Residuen) für jede Vorhersage
# - Ideale Residuen sind zufällig um 0 verteilt
# - Muster in den Residuen deuten auf Probleme hin

# %%

# %%
def plot_residuals():
    plt.figure(figsize=(10, 6))
    plt.scatter(train_predictions, train_residuals, alpha=0.6, label="Training")
    plt.scatter(test_predictions, test_residuals, alpha=0.6, label="Test")
    plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend()
    plt.title("Residual Plot")
    plt.show()


# %%

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

# %% [markdown]
#
# ## Workshop: Pizza-Lieferzeit mit Scikit-Learn
#
# In diesem Workshop werden wir nochmal ein Machine Learning Modell trainieren,
# um die Lieferzeit einer Pizza basierend auf der Entfernung vom Restaurant
# vorherzusagen. Dabei werden wir die Funktionalität von Scikit-Learn nutzen.
#
# **Ziel**: Anwendung aller gelernten Konzepte:
# - Train-Test Split mit `train_test_split()`
# - Training mit `LinearRegression`
# - Evaluation mit MAE, MSE, RMSE und R²
# - Visualisierung der Ergebnisse

# %% [markdown]
#
# Die Daten sind bereits vorbereitet:
#
# - `pizza_distance`: Entfernung vom Restaurant in km
# - `pizza_time`: Lieferzeit in Minuten

# %%
pizza_distance = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
]

# %%
pizza_time = [15, 18, 20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 50]

# %%

# %% [markdown]
#
# ### Aufgabe 1: Daten vorbereiten
#
# Bereiten Sie die Daten für Scikit-Learn vor:
#
# - Konvertieren Sie `pizza_distance` in eine 2D-Liste (Liste von Listen)
# - Speichern Sie das Ergebnis in `X`
# - Speichern Sie `pizza_time` in `y`

# %%

# %% [markdown]
#
# ### Aufgabe 2: Train-Test Split
#
# Teilen Sie die Daten mit `train_test_split()` auf:
#
# - Verwenden Sie ca. 60% der Daten für das Training (40% für Test)
# - Setzen Sie `random_state=42` für Reproduzierbarkeit
# - Speichern Sie die Ergebnisse in `X_train`, `X_test`, `y_train`, `y_test`

# %%

# %% [markdown]
#
# ### Aufgabe 3: Modell trainieren
#
# Trainieren Sie ein lineares Regressionsmodell:
#
# - Erstellen Sie ein `LinearRegression` Modell
# - Trainieren Sie es mit den Trainingsdaten
# - Geben Sie die gelernten Parameter (Koeffizient und Intercept) aus

# %%

# %% [markdown]
#
# ### Aufgabe 4: Vorhersagen machen
#
# Erstellen Sie Vorhersagen für Trainings- und Testdaten:
#
# - Verwenden Sie `model.predict()` für beide Datensätze
# - Speichern Sie die Ergebnisse in `train_predictions` und `test_predictions`

# %%

# %% [markdown]
#
# ### Aufgabe 5: Modell evaluieren
#
# Berechnen Sie die folgenden Metriken für Trainings- und Testdaten:
#
# - MAE (Mean Absolute Error)
# - MSE (Mean Squared Error)
# - RMSE (Root Mean Squared Error)
# - R² Score
#
# Verwenden Sie die entsprechenden Funktionen aus `sklearn.metrics`.

# %%

# %% [markdown]
#
# ### Aufgabe 6: Metriken vergleichen
#
# Erstellen Sie eine übersichtliche Vergleichstabelle der Metriken für
# Trainings- und Testdaten.
#
# **Analysieren Sie die Ergebnisse:**
# - Sind die Metriken ähnlich für Training und Test?
# - Was sagt der R² Score über die Modellqualität aus?
# - Wie groß ist die durchschnittliche Abweichung (MAE)?

# %%

# %% [markdown]
#
# ### Aufgabe 7: Ergebnisse visualisieren
#
# Erstellen Sie einen Plot, der zeigt:
#
# - Die Trainingsdaten als Scatter-Plot (blau)
# - Die Testdaten als Scatter-Plot (orange)
# - Die Modellvorhersagen als rote Linie
#
# Beschriften Sie die Achsen und fügen Sie eine Legende hinzu.

# %%

# %% [markdown]
#
# ### Aufgabe 8: Residuen-Plot
#
# Erstellen Sie einen Residuen-Plot:
#
# - X-Achse: Vorhergesagte Werte
# - Y-Achse: Residuen (Fehler = tatsächlicher Wert - vorhergesagter Wert)
# - Plotten Sie Trainings- und Testdaten
# - Fügen Sie eine horizontale Linie bei y=0 hinzu
#
# **Analysieren Sie:** Sind die Residuen zufällig um 0 verteilt?

# %%

# %% [markdown]
#
# ### Aufgabe 9: Neue Vorhersagen
#
# Verwenden Sie das trainierte Modell, um die Lieferzeit für folgende
# Entfernungen vorherzusagen:
#
# - 2.8 km
# - 5.2 km
# - 9.0 km
# - 10.5 km

# %%

# %% [markdown]
#
# ### Aufgabe 10: Modellqualität einschätzen
#
# **Diskussionsfragen:**
#
# 1. Ist das Modell gut für die Vorhersage geeignet? (Betrachten Sie R²)
# 2. Wie groß ist der durchschnittliche Fehler in Minuten? (Betrachten Sie MAE)
# 3. Sind die Residuen zufällig verteilt oder zeigen sie ein Muster?
# 4. Würden Sie dem Modell für die 10.5 km Vorhersage vertrauen? Warum (nicht)?
# 5. Was bedeutet es, dass Training und Test Metriken sehr ähnlich sind?

# %% [markdown]
#
# **Mögliche Antworten:**
#
# 1. **Modellqualität**: R² nahe 1 (z.B. > 0.95) zeigt, dass das Modell die
#    Daten sehr gut erklärt
# 2. **Durchschnittlicher Fehler**: MAE gibt an, wie viele Minuten die
#    Vorhersage im Durchschnitt abweicht (z.B. ±2 Minuten)
# 3. **Residuen**: Idealerweise zufällig verteilt ohne erkennbares Muster. Hier
#    haben wir aber einen deutlich erkennbaren systematischen Fehler
# 4. **10.5 km Vorhersage**: Vorsicht bei Extrapolation! Das Modell wurde nur
#    bis 8 km trainiert
# 5. **Training ≈ Test**: Das Modell generalisiert gut, kein Overfitting
