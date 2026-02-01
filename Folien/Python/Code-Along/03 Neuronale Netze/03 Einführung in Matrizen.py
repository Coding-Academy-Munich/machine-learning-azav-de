# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in Matrizen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Von einem Datenpunkt zu vielen
#
# Bisher:
# - Ein Vektor = ein Datenpunkt (z.B. ein Haus)
# - Viele Datenpunkte = viele Vektoren
#
# Problem: Wie speichern wir viele Datenpunkte effizient?

# %%
import numpy as np

# %% [markdown]
#
# ### Beispiel: Drei Häuser

# %% [markdown]
#
# Format: [Größe_m², Schlafzimmer, Badezimmer, Alter_Jahre, hat_Garage]

# %%
house1 = np.array([100, 3, 2, 10, 1])
house2 = np.array([180, 4, 3, 2, 1])
house3 = np.array([120, 2, 2, 20, 0])

# %% [markdown]
#
# Drei separate Vektoren - umständlich!

# %% [markdown]
#
# ## Lösung: Matrizen
#
# Eine Matrix ist eine **Tabelle von Zahlen**:
# - Zeilen und Spalten
# - Wie eine Excel-Tabelle
# - Perfekt für viele Datenpunkte!

# %% [markdown]
#
# ### Unsere Häuser als Matrix
#
# Wir stapeln die Vektoren übereinander:

# %%
houses = np.array([
    [100, 3, 2, 10, 1],  # House 1
    [180, 4, 3, 2, 1],   # House 2
    [120, 2, 2, 20, 0]   # House 3
])

# %%

# %% [markdown]
#
# ## Struktur einer Matrix
#
# - **Jede Zeile** = ein Datenpunkt (ein Haus)
# - **Jede Spalte** = ein Feature (Größe, Zimmer, ...)
#
# Das ist die Standard-Organisation in Machine Learning!

# %% [markdown]
#
# ## Matrix-Shape (Form)
#
# Matrizen haben zwei Dimensionen:
# - Anzahl Zeilen (Datenpunkte)
# - Anzahl Spalten (Features)

# %%

# %% [markdown]
#
# `(3, 5)` bedeutet:
# - 3 Zeilen (3 Häuser)
# - 5 Spalten (5 Features)

# %% [markdown]
#
# ### Anzahl der Zeilen und Spalten

# %% [markdown]
#
# Anzahl der Zeilen (Datenpunkte):

# %%

# %% [markdown]
#
# Anzahl der Spalten (Features):

# %%

# %% [markdown]
#
# ## Auf Matrix-Elemente zugreifen
#
# Mit zwei Indizes: `[Zeile, Spalte]`

# %%
houses = np.array([
    [100, 3, 2, 10, 1],  # House 1
    [180, 4, 3, 2, 1],   # House 2
    [120, 2, 2, 20, 0]   # House 3
])

# %% [markdown]
#
# ### Einzelne Elemente

# %% [markdown]
#
# Größe von Haus 1 (Zeile 0, Spalte 0):

# %%

# %% [markdown]
#
# Schlafzimmer von Haus 2 (Zeile 1, Spalte 1):

# %%

# %% [markdown]
#
# Garage von Haus 3 (Zeile 2, Spalte 4):

# %%

# %% [markdown]
#
# ## Ganze Zeilen abrufen
#
# Eine Zeile = alle Features eines Datenpunkts

# %% [markdown]
#
# Alle Features von Haus 1:

# %%

# %% [markdown]
#
# Alle Features von Haus 2:

# %%

# %% [markdown]
#
# Das Ergebnis ist ein Vektor!

# %%

# %% [markdown]
#
# ## Ganze Spalten abrufen
#
# Eine Spalte = ein Feature für alle Datenpunkte

# %% [markdown]
#
# Syntax: `[:, Spalte]`
# - `:` bedeutet "alle Zeilen"

# %% [markdown]
#
# Größe aller Häuser (Spalte 0):

# %%

# %% [markdown]
#
# Schlafzimmer aller Häuser (Spalte 1):

# %%

# %% [markdown]
#
# Garage aller Häuser (Spalte 4):

# %%

# %% [markdown]
#
# ## Mehrere Zeilen oder Spalten
#
# Wir können Slicing verwenden:

# %% [markdown]
#
# Erste zwei Häuser:

# %%

# %% [markdown]
#
# Erste drei Features aller Häuser:

# %%

# %% [markdown]
#
# ## Praktisches Beispiel: Studenten-Daten

# %% [markdown]
#
# Format: [Hausaufgaben_Durchschnitt, Zwischenprüfung, Anwesenheit%, Lernstunden_pro_Woche]

# %%
students = np.array([
    [85, 78, 90, 20],  # Alice
    [92, 88, 95, 30],  # Bob
    [70, 65, 75, 10],  # Charlie
    [88, 82, 85, 25],  # Diana
    [75, 70, 60, 15]   # Eve
])

# %%

# %%
student_names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
feature_names = ["Homework", "Midterm", "Attendance", "Study Hours"]

# %% [markdown]
#
# ### Wie viele Studenten und Features?

# %%

# %% [markdown]
#
# ### Daten eines Studenten

# %% [markdown]
#
# Alle Daten von Bob (Index 1):

# %%

# %%

# %% [markdown]
#
# ### Ein Feature für alle Studenten

# %% [markdown]
#
# Hausaufgaben-Durchschnitte aller Studenten (Spalte 0):

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - Eine Matrix ist eine Tabelle von Zahlen
# - Jede Zeile = ein Datenpunkt
# - Jede Spalte = ein Feature
# - `.shape` gibt (Anzahl Zeilen, Anzahl Spalten)
# - Zugriff mit `[Zeile, Spalte]`
# - Ganze Zeilen: `[Zeile]`
# - Ganze Spalten: `[:, Spalte]`

# %% [markdown]
#
# ## Workshop: Arbeiten mit Matrizen
#
# Probieren Sie es selbst!

# %% [markdown]
#
# ### Aufgabe 1: Matrix erstellen
#
# Erstellen Sie eine Matrix mit Daten von 4 Autos:
# - Auto 1: [25000, 50000, 5, 4]  (Preis, km, Alter, Türen)
# - Auto 2: [18000, 80000, 8, 2]
# - Auto 3: [32000, 30000, 3, 4]
# - Auto 4: [15000, 120000, 10, 4]

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 2: Daten abrufen
#
# Verwenden Sie die Auto-Matrix von oben:
# 1. Was kostet Auto 3?
# 2. Zeigen Sie alle Daten von Auto 2
# 3. Zeigen Sie alle Preise (erste Spalte)
# 4. Zeigen Sie alle Kilometerständeände (zweite Spalte)

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 3: Produktbewertungen
#
# Ein Online-Shop hat Bewertungen für 5 Produkte von 4 Nutzern:

# %% [markdown]
#
# Zeilen = Nutzer, Spalten = Produkte
# Bewertungen von 1 (schlecht) bis 5 (ausgezeichnet)

# %%
ratings = np.array([
    [5, 3, 4, 2, 5],  # User 1
    [4, 4, 3, 3, 4],  # User 2
    [3, 5, 4, 1, 3],  # User 3
    [5, 4, 5, 4, 5]   # User 4
])

# %% [markdown]
#
# 1. Welche Bewertung gab User 3 für Produkt 2 (Index 1)?
# 2. Zeigen Sie alle Bewertungen von User 1
# 3. Zeigen Sie alle Bewertungen für Produkt 5 (Index 4)
# 4. Zeigen Sie die Bewertungen der ersten zwei User

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 4: Wetterdaten
#
# Erstellen Sie eine Matrix mit Wetterdaten für 7 Tage:
# - Spalten: [Temperatur_max, Temperatur_min, Niederschlag_mm, Wind_kmh]
# - Erfinden Sie realistische Werte für eine Woche

# %%
# Your solution here

# %%
