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
predictions

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
houses_matrix

# %%
houses_matrix.shape

# %% [markdown]
#
# Anzahl der Zeilen (Häuser)

# %%
houses_matrix.shape[0]

# %% [markdown]
#
# Anzahl der Spalten (Features)

# %%
houses_matrix.shape[1]  # number of columns (features)

# %% [markdown]
#
# ## Matrix visualisieren

# %%
import matplotlib.pyplot as plt

# %%
def plot_matrix(matrix):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(houses_matrix, cmap="YlOrRd", aspect="auto", vmax=1000)
    feature_names = ["Size", "Beds", "Baths", "Age", "Garage"]
    ax.set_xticks(range(5))
    ax.set_xticklabels(feature_names)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"House {i+1}" for i in range(3)])

    for i in range(3):
        for j in range(5):
            text = ax.text(
                j, i, houses_matrix[i, j], ha="center", va="center", color="black"
            )

    ax.set_title("Houses Matrix")
    plt.tight_layout()
    plt.show()

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
houses_matrix[0, 0]

# %% [markdown]
#
# Haus 2, Schlafzimmer

# %%
houses_matrix[1, 1]

# %% [markdown]
#
# Haus 3, Garage

# %%
houses_matrix[2, 4]

# %% [markdown]
#
# ### Zugriff auf ganze Zeilen (Häuser)
#
# Features von Haus 1

# %%
houses_matrix[0]

# %% [markdown]
#
# Features von Haus 2

# %%
houses_matrix[1]

# %% [markdown]
#
# Zugriff auf ganze Spalten (Features)
#
# Größe aller Häuser

# %%
houses_matrix[:, 0]

# %% [markdown]
#
# Schlafzimmer aller Häuser

# %%
houses_matrix[:, 1]

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
matrix[0] = [10, 20, 30]  # Change first row

# %%
matrix

# %%
matrix[:, 1] = [99, 88]  # Change second column

# %%
matrix

# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])  # Reset

# %%
matrix[0] = matrix[0] + 10  # Add 10 to first row

# %%
matrix[1] = matrix[1] + 20  # Add 20 to second row

# %%
matrix

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
houses_matrix

# %%
coefficients

# %%
houses_matrix @ coefficients

# %%
predictions_vector = houses_matrix @ coefficients + intercept

# %%
predictions_vector

# %%
print("All predictions at once:")
for i, pred in enumerate(predictions_vector):
    print(f"House {i+1}: {pred:.1f}k€")

# %% [markdown]
#
# Vergleiche mit Schleifen-Ergebnissen

# %%
print("\nSame as our loop? ", np.allclose(predictions, predictions_vector))

# %% [markdown]
#
# ## Geschwindigkeitsvergleich

# %% [markdown]
#
# Erstelle viele Häuser

# %%
n_houses = 10000
n_features = 5

# %% [markdown]
#
# Zufällige Hausdaten

# %%
large_houses_matrix = np.random.rand(n_houses, n_features) * 100
large_coefficients = np.random.randn(n_features)

# %% [markdown]
#
# Zeitmessung für Schleifenversion

# %%
start = time.time()
predictions_loop = []
for i in range(n_houses):
    pred = np.dot(large_houses_matrix[i], large_coefficients) + intercept
    predictions_loop.append(pred)
time_loop = time.time() - start

# %% [markdown]
#
# Zeitmessung für Matrix-Version

# %%
start = time.time()
predictions_matrix = large_houses_matrix @ large_coefficients + intercept
time_matrix = time.time() - start

# %%
print(f"Predicting for {n_houses} houses:")
print(f"Loop version: {time_loop:.4f} seconds")
print(f"Matrix version: {time_matrix:.4f} seconds")
print(f"Matrix is {time_loop/time_matrix:.1f}x faster!")

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
matrix

# %%
matrix.shape

# %%
transposed = matrix.T  # oder np.transpose(matrix)

# %%
transposed

# %%
transposed.shape

# %% [markdown]
#
# ## Elementweise Operationen

# %%
matrix = np.array([[10, 20, 30], [40, 50, 60]])

# %% [markdown]
#
# Addieren/Subtrahieren/Multiplizieren mit Skalar

# %%
matrix

# %%
matrix + 5

# %%
matrix * 2

# %%
matrix / 10

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
matrix.sum()

# %% [markdown]
#
# Mittelwert aller Elemente

# %%
matrix.mean()

# %% [markdown]
#
# Standardabweichung aller Elemente (Wie weit streuen die Werte?)

# %%
matrix.std()

# %% [markdown]
#
# Transponieren der Matrix

# %%
matrix.T

# %%
matrix.transpose()

# %% [markdown]
#
# Kopie der Matrix

# %%
matrix_copy = matrix.copy()

# %%
matrix_copy

# %%
matrix_copy[0, 0] = 999

# %%
matrix_copy

# %%
matrix  # Original bleibt unverändert

# %% [markdown]
#
# - Viele Operationen können entlang der Zeilen oder Spalten ausgeführt werden:
#   - .sum(axis=0) für Spaltensummen
#   - .sum(axis=1) für Zeilensummen

# %%
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# %%
matrix

# %% [markdown]
#
# Spaltensummen

# %%
matrix.sum(axis=0)

# %% [markdown]
#
# Zeilensummen

# %%
matrix.sum(axis=1)

# %% [markdown]
#
# Spaltenmittelwert

# %%
matrix.mean(axis=0)

# %% [markdown]
#
# Zeilenmittelwert

# %%
matrix.mean(axis=1)


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
students_matrix = np.array(student_data)

# %%
students_matrix

# %% [markdown]
#
# ### Aufgabe 2: Batch-Vorhersagen
#
# Berechnen Sie alle Abschlussnoten auf einmal:

# %%
final_grades = students_matrix @ grade_coefficients + grade_intercept

# %%
final_grades

# %%
print("Final Grade Predictions:")
print("-" * 30)
for name, grade in zip(student_names, final_grades):
    print(f"{name:10}: {grade:.1f}")

# %% [markdown]
#
# Finde besten und schlechtesten Studenten

# %%
best_idx = final_grades.argmax()
worst_idx = final_grades.argmin()
print(f"\nBest student: {student_names[best_idx]} ({final_grades[best_idx]:.1f})")
print(f"Worst student: {student_names[worst_idx]} ({final_grades[worst_idx]:.1f})")

# %% [markdown]
#
# ### Aufgabe 3: Feature-Statistiken
#
# Berechnen Sie Durchschnitt und Standardabweichung für jedes Feature:

# %%
feature_means = students_matrix.mean(axis=0)  # axis=0 means along rows
feature_stds = students_matrix.std(axis=0)

# %%
print("Feature Statistics:")
print("-" * 40)
for name, mean, std in zip(feature_names, feature_means, feature_stds):
    print(f"{name:12}: μ={mean:.1f}, σ={std:.1f}")

# %% [markdown]
#
# Welches Feature variiert am meisten?

# %%
most_variable_idx = feature_stds.argmax()
print(
    f"\nMost variable feature: {feature_names[most_variable_idx]} (σ={feature_stds[most_variable_idx]:.1f})"
)

# %% [markdown]
#
# ### Aufgabe 4: Was-wäre-wenn-Analyse
#
# Was passiert mit den Noten, wenn alle Studenten:
# 1. 10 Punkte mehr bei der Zwischenprüfung hätten?
# 2. Ihre Anwesenheit um 10% verbessern?

# %% [markdown]
#
# Szenario 1: +10 Punkte bei Zwischenprüfung

# %%
students_scenario1 = students_matrix.copy()
students_scenario1[:, 1] += 10  # Column 1 is midterm
grades_scenario1 = students_scenario1 @ grade_coefficients + grade_intercept

# %%
print("Scenario 1: +10 points on midterm")
for name, original, new in zip(student_names, final_grades, grades_scenario1):
    improvement = new - original
    print(f"{name:10}: {original:.1f} → {new:.1f} (+{improvement:.1f})")

# %% [markdown]
#
# Szenario 2: +10% Anwesenheit

# %%
students_scenario2 = students_matrix.copy()
students_scenario2[:, 2] = np.minimum(100, students_scenario2[:, 2] + 10)  # Cap at 100
grades_scenario2 = students_scenario2 @ grade_coefficients + grade_intercept

# %%
print("\nScenario 2: +10% attendance")
for name, original, new in zip(student_names, final_grades, grades_scenario2):
    improvement = new - original
    print(f"{name:10}: {original:.1f} → {new:.1f} (+{improvement:.1f})")

# %% [markdown]
#
# Welche Verbesserung ist besser?

# %%
avg_improvement_midterm = (grades_scenario1 - final_grades).mean()
avg_improvement_attendance = (grades_scenario2 - final_grades).mean()

# %%
print(f"\nAverage improvement from midterm: {avg_improvement_midterm:.1f}")
print(f"Average improvement from attendance: {avg_improvement_attendance:.1f}")


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
    features_matrix = np.array(features_matrix)
    coefficients = np.array(coefficients)

    # Check dimensions
    if features_matrix.ndim != 2:
        raise ValueError("Features must be a 2D matrix")

    if features_matrix.shape[1] != len(coefficients):
        raise ValueError(
            f"Feature count mismatch: matrix has {features_matrix.shape[1]} features, "
            f"but {len(coefficients)} coefficients provided"
        )

    return features_matrix @ coefficients + intercept


# %% [markdown]
#
# Testen Sie die Funktion

# %%
test_predictions = predict_batch(students_matrix, grade_coefficients, grade_intercept)
print("Test predictions:", test_predictions)
print("Match our earlier results?", np.allclose(test_predictions, final_grades))
