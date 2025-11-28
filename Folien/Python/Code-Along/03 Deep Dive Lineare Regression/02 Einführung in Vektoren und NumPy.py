# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in Vektoren und NumPy</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Warum NumPy?
#
# Erinnern Sie sich an unsere Berechnungen:
# - Viele Multiplikationen und Additionen
# - Schleifen über Listen
# - Fehleranfällig und langsam bei vielen Daten

# %% [markdown]
#
# Unser bisheriger Ansatz

# %%
features = [120, 3, 2, 10, 1]
coefficients = [0.3, 20.0, 15.0, -2.0, 25.0]

# %% [markdown]
#
# Manuelle Berechnung

# %%
sum(f * c for f, c in zip(features, coefficients))

# %% [markdown]
#
# ## NumPy: Numerical Python
#
# - Bibliothek für numerische Berechnungen
# - Macht Mathematik mit vielen Zahlen einfach
# - Viel schneller als normale Python-Listen
# - Standard in Data Science und Machine Learning

# %% [markdown]
#
# ## NumPy installieren und importieren

# %% [markdown]
#
# NumPy ist normalerweise in Data Science Umgebungen vorinstalliert. Falls nicht:

# %%
# !pip install numpy --root-user-action=ignore

# %%
import numpy as np  # 'np' is the standard abbreviation

# %% [markdown]
#
# NumPy Version:

# %%
np.__version__

# %% [markdown]
#
# ## Ihr erstes NumPy Array
#
# - Ein Array ist wie eine Liste, aber optimiert für Mathematik
# - Kann aus Listen erstellt werden
# - Unterstützt viele mathematische Operationen direkt

# %%
my_list = [1, 2, 3, 4, 5]

# %%

# %%

# %%
my_list

# %%

# %%
type(my_list)

# %%

# %% [markdown]
#
# ## Arrays inspizieren

# %%
arr = np.array([10, 20, 30, 40, 50])

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Auf Elemente zugreifen

# %%
arr = np.array([10, 20, 30, 40, 50])

# %% [markdown]
#
# Genau wie Listen!

# %%

# %%

# %%

# %% [markdown]
#
# Aber Vorsicht - Arrays haben einen festen Typ

# %%

# %%

# %%

# %% [markdown]
#
# ## Vektoren: Listen mit Superkräften
#
# In der Mathematik ist ein Vektor eine geordnete Liste von Zahlen:
# - Kann Koordinaten im Raum darstellen
# - Kann Features eines Datenpunkts darstellen
# - Kann mit anderen Vektoren verrechnet werden

# %% [markdown]
#
# ## Vektoren visualisieren (2D)

# %%
import matplotlib.pyplot as plt
from linear_regression_plots import plot_2d_vector, plot_vector_addition

# %% [markdown]
#
# Ein 2D-Vektor als Koordinaten

# %%
vector_2d = np.array([3, 2])

# %% [markdown]
#
# Visualisiere als Pfeil vom Ursprung

# %%

# %% [markdown]
#
# ## Vektoren als Feature-Sammlungen
#
# In ML repräsentieren Vektoren oft Features:
# - Haus-Features als Vektor


# %%
house_vector = np.array([120, 3, 2, 10, 1])  # size, bedrooms, bathrooms, age, garage

# %%

# %%

# %% [markdown]
#
# - Jede Position hat eine Bedeutung

# %%
feature_names = ["size_sqm", "bedrooms", "bathrooms", "age_years", "garage"]

# %%
for i, (name, value) in enumerate(zip(feature_names, house_vector)):
    print(f"Position {i}: {name} = {value}")

# %% [markdown]
#
# ## Einfache Array-Operationen
#
# NumPy macht Mathematik mit Arrays einfach:

# %% [markdown]
#
# ## Skalare Operationen
#
# Operationen mit einzelnen Zahlen:

# %%

# %%

# %% [markdown]
#
# 50 zu allen Preisen addieren

# %%

# %%

# %% [markdown]
#
# 10% Rabatt auf alle Preise

# %%

# %%

# %% [markdown]
#
# ## Vergleich: Listen vs. Arrays
#
# - Listen unterstützen keine direkten mathematischen Operationen
#   - Benötigen Schleifen oder List Comprehensions
# - NumPy Arrays unterstützen elementweise Operationen direkt

# %%
list_prices = [100, 200, 150, 300, 250]

# %%
array_prices = np.array([100, 200, 150, 300, 250])

# %% [markdown]
#
# Mit Listen - Comprehension ist nötig

# %%

# %%

# %% [markdown]
#
# Mit Arrays - einfach multiplizieren!

# %%

# %%

# %% [markdown]
#
# ### Vorsicht: Listen und Multiplikation
#
# - Man kann auch Listen mit ganzen Zahlen multiplizieren
# - Das bedeutet etwas ganz anderes als in der Mathematik!

# %%

# %%


# %% [markdown]
#
# ## Vektoraddition
#
# - Wir können Vektoren addieren
# - Addition erfolgt **elementweise**
# - Die Vektoren müssen die gleiche Länge haben


# %% [markdown]
#
# ### Beispiel
#
# - Monatliche Verkäufe für zwei Produkte
# - Die Summe ist der Gesamtverkauf pro Monat

# %%
product_a = np.array([100, 120, 90, 110, 105])
product_b = np.array([80, 85, 95, 90, 88])

# %%

# %%

# %% [markdown]
#
# Gesamtverkäufe pro Monat

# %%

# %%

# %% [markdown]
#
# ## Visualisierung der Vektoraddition (2D)

# %%
v1 = np.array([3, 1])
v2 = np.array([1, 2])

# %%
v_sum = v1 + v2

# %% [markdown]
#
# Zeichne Vektoren

# %%

# %% [markdown]
#
# ## Methoden auf Vektoren
#
# - NumPy bietet eine Vielzahl von Methoden für Vektoren
# - Häufig verwendete Methoden sind:
#   - `np.sum()` für die Summe aller Elemente
#   - `np.mean()` für den Durchschnitt
#   - `np.max()` und `np.min()` für Maximum und Minimum
#   - `np.argmax()` und `np.argmin()` für die Indizes von Max/Min

# %%

# %% [markdown]
#
# Summe aller Elemente

# %%

# %% [markdown]
#
# Durchschnitt der Elemente

# %%

# %%

# %%

# %% [markdown]
#
# Maximum und Minimum der Elemente

# %%

# %%

# %% [markdown]
#
# Index des Maximums

# %%

# %% [markdown]
#
# Index des Minimums

# %%

# %% [markdown]
#
# ## Logische Operationen auf Vektoren
#
# - Logische Operationen können **elementweise** auf NumPy Arrays angewendet werden.
# - Gängige Operationen sind:
#   - Größer als (`>`)
#   - Kleiner als (`<`)
#   - Gleich (`==`)
#   - Logisches UND (`&`)
#   - Logisches ODER (`|`)

# %%
data = np.array([10, 25, 30, 45, 50])

# %% [markdown]
#
# Ist der jeweilige Wert größer als 30?

# %%

# %%

# %%

# %% [markdown]
#
# Ist der jeweilige Wert gleich 25?

# %%

# %%

# %%

# %% [markdown]
#
# ## Fancy Indexing
#
# - Boolesche Arrays können als Indizes verwendet werden, um Elemente auszuwählen
# - Das Ergebnis enthält nur die Elemente, bei denen der Index `True` ist
# - Nützlich für Filterung und bedingte Auswahl

# %%
data = np.array([10, 25, 30, 45, 50])
data

# %%
mask = data > 30
mask

# %% [markdown]
#
# Werte größer als 30

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - NumPy Arrays sind optimierte Listen für numerische Berechnungen
# - Vektoren sind geordnete Zahlensammlungen
# - Arrays ermöglichen Operationen ohne Schleifen
# - Elementweise Operationen sind einfach und schnell
# - Perfekt für ML-Features und Koeffizienten!

# %% [markdown]
#
# ## Nächste Schritte
#
# - Wir können Features und Koeffizienten als Vektoren darstellen
# - Aber wie berechnen wir effizient Vorhersagen?
# - Nächstes Thema: Das Skalarprodukt (Dot Product)
# - Die wichtigste Operation in linearer Regression!

# %% [markdown]
#
# ## Workshop: NumPy Grundlagen
#
# Üben wir den Umgang mit NumPy Arrays!

# %% [markdown]
#
# ### Aufgabe 1: Arrays erstellen
#
# Erstellen Sie NumPy Arrays für:
# 1. Die Temperaturen einer Woche: [18, 20, 22, 19, 21, 23, 20]
# 2. Die Preise von 5 Produkten: [9.99, 14.50, 7.25, 22.00, 18.75]

# %%

# %% [markdown]
#
# ### Aufgabe 2: Array-Operationen
#
# Mit den Temperatur-Daten:
# 1. Konvertieren Sie alle Temperaturen von Celsius zu Fahrenheit (F = C × 1.8 + 32)
# 2. Erzeugen Sie einen Vektor, der alle Celsius-Werte über 20°C enthält
# 3. Erzeugen Sie einen Vektor, der alle Celsius-Werte für Tage mit über 70°F enthält

# %%
temperatures = np.array([18, 20, 22, 19, 21, 23, 20])

# %%

# %% [markdown]
#
# ### Aufgabe 3: Vektoroperationen
#
# Ein Geschäft verkauft 3 Produkte über 4 Wochen:

# %%
# Sales data (units sold per week)
week1 = np.array([10, 15, 8])
week2 = np.array([12, 14, 10])
week3 = np.array([8, 16, 12])
week4 = np.array([15, 13, 9])

# %% [markdown]
#
# Berechnen Sie den Gesamtverkauf pro Produkt.

# %%

# %% [markdown]
#
# Wie viele Produkte hätte das Geschäft verkauft, wenn der Umsatz 20% höher
# gewesen wäre?

# %%

# %% [markdown]
#
# Erzeugen Sie einen Vektor mit den Verkaufszahlen aller Produkte aus, von denen
# insgesamt weniger als 50 verkauft wurden.

# %%

# %%
