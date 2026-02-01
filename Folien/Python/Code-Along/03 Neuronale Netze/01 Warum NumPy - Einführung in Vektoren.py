# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Warum NumPy? - Einführung in Vektoren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was ist NumPy?
#
# - **Num**erical **Py**thon
# - Eine Bibliothek für Berechnungen mit vielen Zahlen
# - Macht Mathematik mit Daten einfach und schnell
# - Wird in fast allen Data Science und Machine Learning Projekten verwendet

# %% [markdown]
#
# ## Warum brauchen wir NumPy?
#
# Problem mit normalen Python-Listen:
# - Langsam bei vielen Zahlen
# - Mathematik ist umständlich
# - Wir müssen Schleifen schreiben

# %% [markdown]
#
# ### Beispiel: Preise um 10% erhöhen

# %% [markdown]
#
# Mit normalen Python-Listen:

# %%
prices_list = [10, 20, 15, 30, 25]

# %% [markdown]
#
# Wir müssen eine Schleife schreiben:

# %%

# %%

# %% [markdown]
#
# ...oder eine Listen-Komprehension verwenden

# %%

# %% [markdown]
#
# Mit NumPy ist es viel einfacher!

# %%
import numpy as np

# %%

# %% [markdown]
#
# Einfach multiplizieren!

# %%

# %% [markdown]
#
# ## NumPy installieren und importieren

# %% [markdown]
#
# NumPy ist bereits installiert, wenn Sie Anaconda oder eine ähnliche
# Distribution verwenden. Falls nicht:

# %%
# !pip install numpy

# %% [markdown]
#
# Import mit der Standard-Abkürzung:

# %%
import numpy as np

# %% [markdown]
#
# `np` wird praktisch immer als Abkürzung verwendet.

# %% [markdown]
#
# ## Was ist ein Vektor?
#
# Ein Vektor ist eine **geordnete Liste von Zahlen**:
# - Die Reihenfolge ist wichtig
# - Jede Position hat eine Bedeutung
# - Perfekt für Features in Machine Learning!

# %% [markdown]
#
# ### Beispiel: Haus-Features als Vektor
#
# Ein Haus hat verschiedene Eigenschaften:
#
# - Position 0: Größe in m² (120)
# - Position 1: Anzahl Schlafzimmer (3)
# - Position 2: Anzahl Badezimmer (2)
# - Position 3: Alter in Jahren (10)
# - Position 4: Hat Garage (1 = ja, 0 = nein)

# %%
house = [120, 3, 2, 10, 1]

# %% [markdown]
#
# ## NumPy Arrays erstellen
#
# Ein NumPy Array ist wie eine optimierte Liste für Zahlen

# %% [markdown]
#
# ### Aus einer Liste erstellen

# %%
my_list = [1, 2, 3, 4, 5]

# %%

# %% [markdown]
#
# Liste in Array konvertieren:

# %%

# %%

# %% [markdown]
#
# ### Oder direkt erstellen

# %% [markdown]
#
# Array direkt erstellen:

# %%

# %% [markdown]
#
# ### Unterschied zwischen Liste und Array

# %%
my_list = [1, 2, 3, 4, 5]
my_array = np.array([1, 2, 3, 4, 5])

# %%

# %%

# %% [markdown]
#
# Beide sehen ähnlich aus, aber Arrays können viel mehr!

# %% [markdown]
#
# ## Arrays inspizieren
#
# NumPy Arrays haben nützliche Eigenschaften:

# %%
temperatures = np.array([18, 20, 22, 19, 21, 23, 20])

# %%

# %% [markdown]
#
# ### Wie lang?

# %% [markdown]
#
# Funktioniert wie bei Listen:

# %%

# %% [markdown]
#
# ### Wie viele Elemente?

# %% [markdown]
#
# Mit NumPy:

# %%

# %% [markdown]
#
# ### Form des Arrays (Shape)

# %%

# %% [markdown]
#
# `(7,)` bedeutet: 7 Elemente, eindimensional

# %% [markdown]
#
# ### Datentyp der Elemente

# %%

# %% [markdown]
#
# `int64` = ganze Zahlen (Integers)

# %%
prices = np.array([9.99, 14.50, 7.25])

# %%

# %% [markdown]
#
# `float64` = Dezimalzahlen (Floats)

# %% [markdown]
#
# ## Auf Elemente zugreifen
#
# Wie bei Listen: mit Index in eckigen Klammern

# %%
temperatures = np.array([18, 20, 22, 19, 21, 23, 20])

# %% [markdown]
#
# ### Einzelne Elemente

# %% [markdown]
#
# Erstes Element (Index 0):

# %%

# %% [markdown]
#
# Drittes Element (Index 2):

# %%

# %% [markdown]
#
# Letztes Element:

# %%

# %% [markdown]
#
# ### Mehrere Elemente (Slicing)

# %% [markdown]
#
# Erste 3 Elemente (Index 0, 1, 2):

# %%

# %% [markdown]
#
# Elemente ab Index 2 bis zum Ende:

# %%

# %% [markdown]
#
# Elemente bis Index 4 (nicht einschließlich):

# %%

# %% [markdown]
#
# ## Praktisches Beispiel: Haus-Features

# %%
# [size_sqm, bedrooms, bathrooms, age_years, has_garage]
house = np.array([120, 3, 2, 10, 1])

# %% [markdown]
#
# Namen für die Features:

# %%
feature_names = ["Size (sqm)", "Bedrooms", "Bathrooms", "Age (years)", "Garage"]

# %% [markdown]
#
# ### Einzelne Features abrufen

# %% [markdown]
#
# Größe des Hauses:

# %%

# %% [markdown]
#
# Anzahl Schlafzimmer:

# %%

# %% [markdown]
#
# Hat eine Garage?

# %%

# %% [markdown]
#
# ### Alle Features anzeigen

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - NumPy macht Berechnungen mit vielen Zahlen einfach
# - Ein Vektor ist eine geordnete Liste von Zahlen
# - NumPy Arrays sind optimierte Listen für Zahlen
# - Arrays haben Eigenschaften: `.size`, `.shape`, `.dtype`
# - Zugriff wie bei Listen: `array[index]`

# %% [markdown]
#
# ## Workshop: Erste Schritte mit NumPy
#
# Probieren Sie es selbst!

# %% [markdown]
#
# ### Aufgabe 1: Array erstellen
#
# Erstellen Sie einen NumPy Array mit den Altersangaben von 5 Personen:
# - Person A: 25 Jahre
# - Person B: 32 Jahre
# - Person C: 18 Jahre
# - Person D: 45 Jahre
# - Person E: 28 Jahre

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 2: Array inspizieren
#
# Finden Sie heraus:
# 1. Wie viele Personen sind im Array?
# 2. Was ist die Form (Shape) des Arrays?
# 3. Was ist der Datentyp?

# %% [markdown]
#
# ### Aufgabe 3: Elemente abrufen
#
# 1. Wie alt ist Person C?
# 2. Wie alt ist die letzte Person?
# 3. Zeigen Sie die Alter der ersten 3 Personen

# %% [markdown]
#
# ### Aufgabe 4: Produkt-Features
#
# Ein Online-Shop speichert Produktinformationen als Vektor:
# [Preis, Bewertung (1-5), Anzahl Verkäufe, Lagerbestand]
#
# Erstellen Sie einen Vektor für ein Produkt mit:
# - Preis: 29.99€
# - Bewertung: 4 Sterne
# - 150 Verkäufe
# - 47 Stück auf Lager

# %%
# Your solution here

# %%
