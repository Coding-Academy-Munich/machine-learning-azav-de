# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Rechnen mit Matrizen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bereits können
#
# - Matrizen erstellen
# - Form inspizieren (`.shape`)
# - Elemente, Zeilen und Spalten abrufen
#
# Jetzt lernen wir: **Rechnen mit Matrizen!**

# %%
import numpy as np

# %% [markdown]
#
# ## Skalare Operationen auf Matrizen
#
# Wie bei Vektoren: Operation wird auf alle Elemente angewendet

# %%
prices = np.array([
    [10, 20, 15],
    [30, 25, 18],
    [12, 22, 28]
])

# %%

# %% [markdown]
#
# ### Alle Preise erhöhen

# %% [markdown]
#
# 5 zu allen Preisen addieren:

# %%

# %% [markdown]
#
# 10% Erhöhung:

# %%

# %% [markdown]
#
# 20% Rabatt:

# %%

# %% [markdown]
#
# ## Elementweise Operationen
#
# Zwei Matrizen gleicher Größe elementweise verrechnen:

# %% [markdown]
#
# Verkäufe Woche 1:

# %%
week1 = np.array([
    [100, 120, 90],
    [110, 105, 115],
    [95, 130, 100]
])

# %% [markdown]
#
# Verkäufe Woche 2:

# %%
week2 = np.array([
    [105, 115, 95],
    [120, 110, 108],
    [98, 125, 105]
])

# %% [markdown]
#
# ### Gesamtverkäufe über beide Wochen

# %%

# %%

# %% [markdown]
#
# Position [0, 0]: 100 + 105 = 205
# Position [1, 2]: 115 + 108 = 223
# usw.

# %% [markdown]
#
# ## Aggregationen auf der ganzen Matrix
#
# Alle Werte zu einer Zahl zusammenfassen:

# %%
data = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

# %%

# %% [markdown]
#
# ### Summe aller Elemente

# %%

# %% [markdown]
#
# 10 + 20 + 30 + 40 + 50 + 60 = 210

# %% [markdown]
#
# ### Durchschnitt aller Elemente

# %%

# %% [markdown]
#
# 210 / 6 = 35

# %% [markdown]
#
# ### Minimum und Maximum

# %%

# %%

# %% [markdown]
#
# ## Aggregationen entlang einer Achse
#
# **Achse 0** = entlang der Zeilen (↓) = für jede Spalte
# **Achse 1** = entlang der Spalten (→) = für jede Zeile

# %%
data = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

# %%

# %% [markdown]
#
# ### Summe pro Spalte (axis=0)
#
# Für jede Spalte: addiere alle Zeilen

# %%

# %% [markdown]
#
# - Spalte 0: 10 + 40 = 50
# - Spalte 1: 20 + 50 = 70
# - Spalte 2: 30 + 60 = 90

# %% [markdown]
#
# ### Summe pro Zeile (axis=1)
#
# Für jede Zeile: addiere alle Spalten

# %%

# %% [markdown]
#
# - Zeile 0: 10 + 20 + 30 = 60
# - Zeile 1: 40 + 50 + 60 = 150

# %% [markdown]
#
# ## Praktisches Beispiel: Studenten-Noten

# %% [markdown]
#
# Zeilen = Studenten, Spalten = Prüfungen

# %%
grades = np.array([
    [85, 78, 90],  # Alice
    [92, 88, 95],  # Bob
    [70, 65, 75],  # Charlie
    [88, 82, 85]   # Diana
])

# %%

# %%
student_names = ["Alice", "Bob", "Charlie", "Diana"]
exam_names = ["Exam 1", "Exam 2", "Exam 3"]

# %% [markdown]
#
# ### Durchschnitt pro Student (über alle Prüfungen)

# %% [markdown]
#
# axis=1: für jede Zeile (Student)

# %%

# %%

# %%

# %% [markdown]
#
# ### Durchschnitt pro Prüfung (über alle Studenten)

# %% [markdown]
#
# axis=0: für jede Spalte (Prüfung)

# %%

# %%

# %%

# %% [markdown]
#
# ### Wer hat die beste Durchschnittsnote?

# %%

# %%

# %% [markdown]
#
# ### Welche Prüfung war am schwersten?
#
# Die mit dem niedrigsten Durchschnitt

# %%

# %%

# %% [markdown]
#
# ## ML-Anwendung: Feature-Statistiken
#
# In Machine Learning analysieren wir oft Features:

# %% [markdown]
#
# Format: [Größe_m², Schlafzimmer, Alter_Jahre, Preis_k€]

# %%
houses = np.array([
    [100, 3, 10, 250],
    [180, 4, 2, 450],
    [120, 2, 20, 180],
    [150, 3, 5, 320],
    [90, 2, 15, 200]
])

# %%

# %%
feature_names = ["Size (sqm)", "Bedrooms", "Age (years)", "Price (k€)"]

# %% [markdown]
#
# ### Durchschnitt pro Feature

# %% [markdown]
#
# axis=0: für jede Spalte (Feature)

# %%

# %%

# %% [markdown]
#
# ### Minimum und Maximum pro Feature

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Skalare Operationen**: Eine Zahl mit ganzer Matrix verrechnen
# - **Elementweise Operationen**: Zwei Matrizen elementweise verrechnen
# - **Aggregationen ohne Achse**: Alle Werte → eine Zahl
# - **Aggregationen mit Achse**:
#   - `axis=0`: Pro Spalte (über alle Zeilen)
#   - `axis=1`: Pro Zeile (über alle Spalten)

# %% [markdown]
#
# ## Warum ist das für ML wichtig?
#
# - Wir analysieren oft viele Datenpunkte gleichzeitig
# - Feature-Statistiken helfen uns, Daten zu verstehen
# - Normalisierung und Skalierung basieren auf diesen Operationen
# - Effiziente Berechnungen = schnelleres Training!

# %% [markdown]
#
# ## Workshop: Rechnen mit Matrizen
#
# Probieren Sie es selbst!

# %% [markdown]
#
# ### Aufgabe 1: Verkaufsdaten analysieren
#
# Ein Geschäft hat Verkaufsdaten für 3 Produkte über 4 Wochen:

# %% [markdown]
#
# Zeilen = Wochen, Spalten = Produkte

# %%
sales = np.array([
    [120, 85, 95],   # Week 1
    [130, 90, 88],   # Week 2
    [115, 95, 92],   # Week 3
    [140, 88, 100]   # Week 4
])

# %% [markdown]
#
# 1. Was ist der Gesamtverkauf über alle Wochen und Produkte?
# 2. Was ist der durchschnittliche Verkauf pro Woche (über alle Produkte)?
# 3. Was ist der Gesamtverkauf pro Produkt (über alle Wochen)?
# 4. Welches Produkt wurde am meisten verkauft (insgesamt)?

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 2: Temperaturdaten
#
# Temperaturen (°C) für 5 Städte über 7 Tage:

# %% [markdown]
#
# Zeilen = Tage, Spalten = Städte

# %%
temperatures = np.array([
    [22, 18, 25, 20, 19],  # Monday
    [24, 19, 27, 21, 20],  # Tuesday
    [23, 20, 26, 22, 21],  # Wednesday
    [20, 17, 23, 19, 18],  # Thursday
    [18, 15, 21, 17, 16],  # Friday
    [19, 16, 22, 18, 17],  # Saturday
    [21, 18, 24, 20, 19]   # Sunday
])

# %% [markdown]
#
# 1. Was ist die durchschnittliche Temperatur pro Stadt (über die Woche)?
# 2. Was ist die durchschnittliche Temperatur pro Tag (über alle Städte)?
# 3. Welche Stadt hatte die höchste Durchschnittstemperatur?
# 4. An welchem Tag war es im Durchschnitt am wärmsten?

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 3: Feature-Normalisierung
#
# In ML normalisieren wir oft Features auf den Bereich 0-1:

# %% [markdown]
#
# Format: [Größe_m², Preis_k€]

# %%
houses = np.array([
    [100, 250],
    [180, 450],
    [120, 180],
    [150, 320],
    [90, 200]
])

# %% [markdown]
#
# Normalisieren Sie jede Spalte (Feature) separat:
# - Finden Sie Min und Max für jedes Feature
# - Formel: `(value - min) / (max - min)`
#
# Tipp: Verwenden Sie `axis=0` für Min und Max

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 4: Budget-Analyse
#
# Ein Unternehmen hat Ausgaben für 4 Abteilungen über 3 Quartale:

# %% [markdown]
#
# Zeilen = Quartale, Spalten = Abteilungen (in k€)

# %%
expenses = np.array([
    [120, 80, 150, 90],   # Q1
    [130, 85, 160, 95],   # Q2
    [125, 82, 155, 88]    # Q3
])

# %% [markdown]
#
# 1. Gesamtausgaben pro Quartal
# 2. Durchschnittliche Ausgaben pro Abteilung
# 3. Budget um 10% für alle Abteilungen erhöhen - wie sieht die neue Matrix aus?
# 4. Welche Abteilung hatte die höchsten Gesamtausgaben über alle Quartale?

# %%
# Your solution here

# %%
