# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Von Einem zu Vielen - Einführung in Matrizen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Das Problem: Viele Vorhersagen
#
# Bisher: Eine Vorhersage nach der anderen
# - Ein Feature-Vektor → Eine Vorhersage
#
# Realität: Wir brauchen oft viele Vorhersagen
# - 1000 Häuser bewerten
# - Alle Testdaten vorhersagen

# %%
import numpy as np
import time

# %% [markdown]
#
# Hauspreismodell

# %%
coefficients = np.array([0.5, 25.0, 20.0, -1.5, 30.0])
intercept = 100.0

# %% [markdown]
#
# Drei Häuser zum Vorhersagen

# %%
house1 = np.array([100, 3, 1, 10, 0])
house2 = np.array([180, 4, 3, 2, 1])
house3 = np.array([120, 2, 2, 20, 1])

# %% [markdown]
#
# ## Lösung 1: Schleife
#
# Wir könnten eine Schleife verwenden:

# %%
houses = [house1, house2, house3]
predictions = []

# %%
for house in houses:
    pred = np.dot(house, coefficients) + intercept
    predictions.append(pred)

# %%

# %% [markdown]
#
# ## Problem mit Schleifen
#
# - Langsam bei vielen Daten
# - Verliert einen Teil von NumPy's Geschwindigkeitsvorteil
# - Nicht elegant
#
# Es gibt einen besseren Weg!

# %% [markdown]
#
# ## Matrizen: Tabellen von Zahlen
#
# Eine Matrix ist ein 2D-Array:
# - Zeilen und Spalten
# - Wie eine Tabelle oder Spreadsheet
# - Perfekt für mehrere Datenpunkte

# %% [markdown]
#
# ## Unsere Häuser als Matrix
#
# - Stapele Häuser zu einer Matrix
# - Jede Zeile ist ein Haus
# - Jede Spalte ist ein Feature


# %%
houses_matrix = np.array(
    [
        [100, 3, 1, 10, 0],  # House 1
        [180, 4, 3, 2, 1],  # House 2
        [120, 2, 2, 20, 1],  # House 3
    ]
)

# %%

# %%

# %% [markdown]
#
# Anzahl der Zeilen (Häuser)

# %%

# %% [markdown]
#
# Anzahl der Spalten (Features)

# %%

# %% [markdown]
#
# ## Matrix visualisieren

# %%
from linear_regression_plots import plot_matrix

# %%
plot_matrix(houses_matrix)

# %% [markdown]
#
# ## Auf Matrix-Elemente zugreifen
#
# - Matrizen haben zwei Indizes: [Zeile, Spalte]
# - Wir können einzelne Elemente, ganze Zeilen oder Spalten auswählen

# %% [markdown]
#
# ### Zugriff auf spezifische Elemente
#
# Haus 1, Größe

# %%

# %% [markdown]
#
# Haus 2, Schlafzimmer

# %%

# %% [markdown]
#
# Haus 3, Garage

# %%

# %% [markdown]
#
# ### Zugriff auf ganze Zeilen (Häuser)
#
# Features von Haus 1

# %%

# %% [markdown]
#
# Features von Haus 2

# %%

# %% [markdown]
#
# ### Zugriff auf ganze Spalten (Features)
#
# Größe aller Häuser

# %%

# %% [markdown]
#
# Schlafzimmer aller Häuser

# %%

# %% [markdown]
#
# ### Änderung von Zeilen oder Spalten
#
# - Wir können ganze Zeilen oder Spalten in der Matrix ändern!
# - Diese Änderungen sind destruktiv (ändern die Originalmatrix)

# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# %%
matrix

# %%

# %%
# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# %%

# %%

# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])  # Reset

# %%

# %%

# %%

# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])  # Reset

# %%

# %%

# %%

# %% [markdown]
#
# ## Elementweise Operationen

# %%
matrix = np.array([[10, 20, 30], [40, 50, 60]])

# %% [markdown]
#
# Addieren/Subtrahieren/Multiplizieren mit Skalar

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Matrix-Vektor-Multiplikation
#
# Die Magie: Wir können eine Matrix mit einem Vektor multiplizieren!
# - Jede Zeile wird mit dem Vektor skalar-multipliziert
# - Ergebnis: Ein Vektor mit allen Vorhersagen

# %% [markdown]
#
# ## Visualisierung der Matrix-Vektor-Multiplikation
#
# Einfaches Beispiel zur Visualisierung

# %%
simple_matrix = np.array([[1, 2], [3, 4], [5, 6]])

# %%
simple_vector = np.array([10, 20])

# %%
simple_matrix

# %%
simple_vector

# %% [markdown]
#
# Manuelle Berechnung

# %%
print("\nManual calculation:")
for i, row in enumerate(simple_matrix):
    dot_product = np.dot(row, simple_vector)
    print(
        f"Row {i}: {row} · {simple_vector} = {row[0]}×{simple_vector[0]} + {row[1]}×{simple_vector[1]} = {dot_product}"
    )

# %% [markdown]
#
# NumPy-Berechnung

# %%
simple_matrix @ simple_vector

# %% [markdown]
#
# ## Batch-Vorhersagen mit Matrix-Multiplikation
#
# - Wir können mehrere Vektoren in eine Matrix packen
# - Dann können wir alle Vorhersagen auf einmal berechnen!

# %%
houses_matrix = np.array([[100, 3, 1, 10, 0], [180, 4, 3, 2, 1], [120, 2, 2, 20, 1]])
coefficients = np.array([0.5, 25.0, 20.0, -1.5, 30.0])
intercept = 100.0

# %%

# %%

# %%

# %% [markdown]
#
# Vergleiche mit Schleifen-Ergebnissen

# %%

# %% [markdown]
#
# ## Geschwindigkeitsvergleich

# %% [markdown]
#
# Erstelle viele Häuser

# %%
n_houses = 100_000
n_features = 5

# %% [markdown]
#
# Zufällige Hausdaten

# %%
large_houses_matrix = np.random.rand(n_houses, n_features) * 100

# %%

# %% [markdown]
#
# Zeitmessung für Schleifenversion

# %%
start = time.time()
predictions_loop = []
for i in range(n_houses):
    pred = np.dot(large_houses_matrix[i], coefficients) + intercept
    predictions_loop.append(pred)
time_loop = time.time() - start

# %% [markdown]
#
# Zeitmessung für Matrix-Version

# %%
start = time.time()
predictions_matrix = large_houses_matrix @ coefficients + intercept
time_matrix = time.time() - start
time_matrix = 1e-8 if time_matrix == 0 else time_matrix  # avoid division by zero on fast systems

# %%

# %% [markdown]
#
# ## Weitere Matrix-Operationen

# %% [markdown]
#
# ## Transponieren
#
# Zeilen und Spalten vertauschen:

# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Methoden auf Matrizen
#
# - Die uns bekannten NumPy-Methoden funktionieren auch auf Matrizen
#   - `.shape` für Dimensionen
#   - `.sum()` für Summen
#   - `.mean()` für Mittelwerte
#   - `.std()` für Standardabweichungen
#   - `.T` für Transponieren
#   - `.copy()` für Kopien

# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# %% [markdown]
#
# Summe aller Elemente

# %%

# %% [markdown]
#
# Mittelwert aller Elemente

# %%

# %% [markdown]
#
# Standardabweichung aller Elemente (Wie weit streuen die Werte?)

# %%

# %% [markdown]
#
# Transponieren der Matrix

# %%

# %%

# %% [markdown]
#
# Kopie der Matrix

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# - Viele Operationen können entlang der Zeilen oder Spalten ausgeführt werden:
#   - .sum(axis=0) für Spaltensummen
#   - .sum(axis=1) für Zeilensummen

# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# %%

# %% [markdown]
#
# Spaltensummen

# %%

# %% [markdown]
#
# Zeilensummen

# %%

# %% [markdown]
#
# Spaltenmittelwert

# %%

# %% [markdown]
#
# Zeilenmittelwert

# %%


# %% [markdown]
#
# ## Zusammenfassung
#
# - Matrizen sind 2D-Arrays (Tabellen von Zahlen)
# - Jede Zeile = ein Datenpunkt
# - Jede Spalte = ein Feature
# - Matrix @ Vektor = Alle Vorhersagen auf einmal!
# - Viel schneller als Schleifen

# %% [markdown]
#
# ## Nächste Schritte
#
# - Wir können jetzt effizient viele Vorhersagen machen
# - Zeit, unser eigenes Regressionsmodell zu bauen!
# - Nächstes Thema: Eine komplette LinearRegression-Klasse
# - Mit fit() und predict() wie Scikit-Learn!

# %% [markdown]
#
# ## Workshop: Studentennoten-Vorhersage
#
# Wir werden folgendes Modell zur Vorhersage von Abschlussnoten verwenden:

# %%
feature_names = ["Homework Average", "Midterm Score", "Attendance", "Study Hours"]

# %%
grade_coefficients = np.array([0.2, 0.4, 0.2, 0.05])
grade_intercept = 10.0

# %%
print("Grade Prediction Model:")
for name, coef in zip(feature_names, grade_coefficients):
    print(f"{name:17}: {coef}")
print(f"{'Base':17}: {grade_intercept}")

# %% [markdown]
#
# ### Aufgabe 1: Matrix erstellen
#
# Erstellen Sie eine Matrix für diese 5 Studenten:

# %%
student_data = [
    [85, 78, 90, 20],  # Student A
    [92, 88, 95, 30],  # Student B
    [70, 65, 75, 10],  # Student C
    [88, 82, 85, 25],  # Student D
    [75, 70, 60, 15],  # Student E
]

# %%
student_names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]


# %%

# %% [markdown]
#
# ### Aufgabe 2: Batch-Vorhersagen
#
# Berechnen Sie alle Abschlussnoten auf einmal:

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 3: Feature-Statistiken
#
# Berechnen Sie Durchschnitt und Standardabweichung für jedes Feature:

# %%

# %% [markdown]
#
# ### Aufgabe 4: Was-wäre-wenn-Analyse
#
# Was passiert mit den Noten, wenn alle Studenten:
# 1. 10 Punkte mehr bei der Zwischenprüfung hätten?
# 2. Ihre Anwesenheit um 10% verbessern?

# %%

# %% [markdown]
#
# ### Aufgabe 5: Batch-Vorhersagefunktion

# %%
def predict_batch(features_matrix, coefficients, intercept):
    """
    Make predictions for multiple samples.

    Parameters:
    - features_matrix: 2D array, each row is one sample
    - coefficients: 1D array of model coefficients
    - intercept: scalar intercept

    Returns:
    - predictions: 1D array of predictions
    """
    # Your solution here
    pass

# %%

# %%
