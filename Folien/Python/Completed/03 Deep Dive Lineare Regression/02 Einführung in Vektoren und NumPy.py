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
result = 0
for f, c in zip(features, coefficients):
    result += f * c

# %%
result

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
my_array = np.array(my_list)

# %%
my_array = np.array([1, 2, 3, 4, 5])

# %%
my_list

# %%
my_array

# %%
type(my_list)

# %%
type(my_array)

# %% [markdown]
#
# ## Arrays inspizieren

# %%
arr = np.array([10, 20, 30, 40, 50])

# %%
arr

# %%
len(arr)

# %%
arr.shape  # Dimensions of the array

# %%
arr.size  # Total number of elements

# %%
arr.dtype  # Type of elements

# %% [markdown]
#
# ## Auf Elemente zugreifen

# %%
arr = np.array([10, 20, 30, 40, 50])

# %% [markdown]
#
# Genau wie Listen!

# %%
arr[0]

# %%
arr[-1]

# %%
arr[1:4]

# %% [markdown]
#
# Aber Vorsicht - Arrays haben einen festen Typ

# %%
arr[0] = 15.5  # Float wird zu Int konvertiert

# %%
arr

# %%
arr.dtype  # Noch immer Integer

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
from plot_utils import plot_2d_vector, plot_vector_addition

# %% [markdown]
#
# Ein 2D-Vektor als Koordinaten

# %%
vector_2d = np.array([3, 2])

# %% [markdown]
#
# Visualisiere als Pfeil vom Ursprung

# %%
plot_2d_vector(vector_2d)

# %% [markdown]
#
# ## Vektoren als Feature-Sammlungen
#
# In ML repräsentieren Vektoren oft Features:
# - Haus-Features als Vektor


# %%
house_vector = np.array([120, 3, 2, 10, 1])  # size, bedrooms, bathrooms, age, garage

# %%
house_vector

# %%
len(house_vector)

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
prices = np.array([100, 200, 150, 300, 250])

# %%
prices

# %% [markdown]
#
# 50 zu allen Preisen addieren

# %%
increased = prices + 50

# %%
increased

# %% [markdown]
#
# 10% Rabatt auf alle Preise

# %%
discounted = prices * 0.9

# %%
discounted

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
list_discounted = [price  * 0.9 for price in list_prices]

# %%
list_discounted

# %% [markdown]
#
# Mit Arrays - einfach multiplizieren!

# %%
array_discounted = array_prices * 0.9

# %%
array_discounted

# %% [markdown]
#
# ### Vorsicht: Listen und Multiplikation
#
# - Man kann auch Listen mit ganzen Zahlen multiplizieren
# - Das bedeutet etwas ganz anderes als in der Mathematik!

# %%
[1, 2, 3] * 3  # List replication, not element-wise multiplication

# %%
np.array([1, 2, 3]) * 3  # Element-wise multiplication


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
product_a

# %%
product_b

# %% [markdown]
#
# Gesamtverkäufe pro Monat

# %%
total_sales = product_a + product_b

# %%
total_sales

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
plot_vector_addition(v1, v2)

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
data = np.array([25, 10, 30, 50, 45])

# %% [markdown]
#
# Summe aller Elemente

# %%
data.sum()

# %% [markdown]
#
# Durchschnitt der Elemente

# %%
data.mean()

# %%
data.sum() / len(data)

# %%
data.sum() / data.size

# %% [markdown]
#
# Maximum und Minimum der Elemente

# %%
data.max()

# %%
data.min()

# %% [markdown]
#
# Index des Maximums

# %%
data.argmax()

# %% [markdown]
#
# Index des Minimums

# %%
data.argmin()



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
data > 30

# %%
big_value_mask = data > 30

# %%
big_value_mask

# %% [markdown]
#
# Ist der jeweilige Wert gleich 25?

# %%
data == 25

# %%
equal_25_mask = data == 25

# %%
equal_25_mask | big_value_mask  # Werte > 30 oder == 25

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
data[mask]

# %%
data[data > 30]  # Direktes Filtern ohne Zwischenspeicherung

# %%
data

# %%
data % 15

# %%
data % 15 == 0

# %%
data[data % 15 == 0]  # Werte, die durch 15 teilbar sind



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

# %% [markdown]
#
# 1. Wochentemperaturen

# %%
temperatures = np.array([18, 20, 22, 19, 21, 23, 20])

# %%
print(f"Temperatures: {temperatures}")
print(f"Shape: {temperatures.shape}")
print(f"Average temperature: {temperatures.mean():.1f}°C")

# %% [markdown]
#
# 2. Produktpreise

# %%
prices = np.array([9.99, 14.50, 7.25, 22.00, 18.75])

# %%
print(f"Prices: {prices}")
print(f"Total price: ${prices.sum():.2f}")

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

# %% [markdown]
#
# 1. Nach Fahrenheit konvertieren

# %%
fahrenheit = temperatures * 1.8 + 32

# %%
print("Celsius to Fahrenheit conversion:")
for c, f in zip(temperatures, fahrenheit):
    print(f"{c}°C = {f:.1f}°F")

# %% [markdown]
#
# 2. Tage über 20°C

# %%
warm_days = temperatures > 20


# %%
print(f"\nTemperatures:    {temperatures}")
print(f"Above 20°C Mask: {warm_days}")
print(f"Above 20°C:      {temperatures[warm_days]}")

# %%
fahrenheit_above_70 = (temperatures * 1.8 + 32) > 70
print(f"Above 70°F Mask: {fahrenheit_above_70}")
print(f"Above 70°F:      {temperatures[fahrenheit_above_70]}")

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

# Prices per unit
prices = np.array([25.0, 40.0, 15.0])

# %% [markdown]
#
# Berechnen Sie:
# 1. Gesamtverkäufe pro Produkt über alle Wochen
# 2. Umsatz pro Woche

# %% [markdown]
#
# 1. Gesamtverkäufe pro Produkt

# %%
total_per_product = week1 + week2 + week3 + week4

# %%
print("Total sales per product:")
product_names = ["Product A", "Product B", "Product C"]
for name, total in zip(product_names, total_per_product):
    print(f"{name}: {total} units")

# %% [markdown]
#
# 2. Umsatz pro Woche

# %%
revenue_week1 = week1 * prices
revenue_week2 = week2 * prices
revenue_week3 = week3 * prices
revenue_week4 = week4 * prices

# %%
print("\nRevenue per week:")
print(f"Week 1: ${revenue_week1.sum():.2f}")
print(f"Week 2: ${revenue_week2.sum():.2f}")
print(f"Week 3: ${revenue_week3.sum():.2f}")
print(f"Week 4: ${revenue_week4.sum():.2f}")

# %%
total_revenue = (revenue_week1 + revenue_week2 + revenue_week3 + revenue_week4).sum()
print(f"\nTotal revenue: ${total_revenue:.2f}")

# %% [markdown]
#
# ### Aufgabe 4: Feature-Vektoren
#
# Erstellen Sie Feature-Vektoren für 3 Wohnungen und berechnen Sie eine
# Bewertung mit gegebenen Gewichten.
#
# |           | Größe (qm) | Schlafzimmer | Etage | Balkon   |
# |-----------|------------|--------------|-------|----------|
# | Wohnung 1 | 65         | 2            | 3     | Ja (1)   |
# | Wohnung 2 | 85         | 3            | 1     | Nein (0) |
# | Wohnung 3 | 55         | 1            | 5     | Ja (1)   |
#
# Die Gewichte sind:
#
# | Feature      | Gewicht |
# |--------------|---------|
# | Größe        | 2.0     |
# | Schlafzimmer | 20.0    |
# | Etage        | 5.0     |
# | Balkon       | 15.0    |

# %%
apt1 = np.array([65, 2, 3, 1])
apt2 = np.array([85, 3, 1, 0])
apt3 = np.array([55, 1, 5, 1])

# %%
print("Apartments:")
print(f"Apt 1: {apt1}")
print(f"Apt 2: {apt2}")
print(f"Apt 3: {apt3}")

# %%
weights = np.array([2.0, 20.0, 5.0, 15.0])

# %%
print(f"\nWeights: {weights}")

# %% [markdown]
#
# Berechnen Sie die Bewertung für jede Wohnung (ohne Schleifen für die
# Berechnung der einzelnen Bewertungen!).

# %%
score1 = (apt1 * weights).sum()
score2 = (apt2 * weights).sum()
score3 = (apt3 * weights).sum()

# %%
print("Apartment Scores:")
print(f"Apartment 1: {score1:.1f} points")
print(f"Apartment 2: {score2:.1f} points")
print(f"Apartment 3: {score3:.1f} points")

# %% [markdown]
#
# Finden Sie die beste Wohnung (die mit der höchsten Bewertung).

# %%
scores = np.array([score1, score2, score3])
best_idx = scores.argmax()
print(f"\nBest apartment: Apartment {best_idx + 1} with {scores[best_idx]:.1f} points")

# %%
