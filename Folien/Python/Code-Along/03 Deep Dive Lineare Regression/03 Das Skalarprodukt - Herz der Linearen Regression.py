# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Das Skalarprodukt - Herz der Linearen Regression</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# Erinnern Sie sich an unsere Vorhersageberechnung:
# 1. Feature × Koeffizient für jedes Feature
# 2. Alle Produkte addieren
# 3. Intercept addieren

# %% [markdown]
#
# ### Unser vertrautes Beispiel
#
# (Im Moment ohne Intercept)

# %%
features = [5.0, 7.0, 3.0]  # distance, traffic, pizzas
coefficients = [2.0, 1.5, 0.5]

# %% [markdown]
#
# Manuelle Berechnung

# %%
result = 0
for f, c in zip(features, coefficients):
    result += f * c
    print(f"{f} × {c} = {f * c}")

# %%
result

# %% [markdown]
#
# ### Dieses Muster hat einen Namen!
#
# **Skalarprodukt** (Dot Product):
# - Multipliziere entsprechende Elemente
# - Addiere alle Produkte
# - Ergebnis ist eine einzelne Zahl (Skalar)

# %% [markdown]
#
# ### Das Skalarprodukt verstehen
#
# Für zwei Vektoren a und b:
# - a = [a₁, a₂, a₃]
# - b = [b₁, b₂, b₃]
# - a · b = a₁×b₁ + a₂×b₂ + a₃×b₃

# %% [markdown]
#
# ### Beispiel: Schritt für Schritt

# %% [markdown]
#
# Zwei Vektoren

# %%
a = [2, 3, 4]
b = [5, 1, 2]

# %% [markdown]
#
# Berechne Skalarprodukt manuell

# %%
dot_product = 0
print("Calculating dot product:")
for i in range(len(a)):
    product = a[i] * b[i]
    dot_product += product
    print(f"Position {i}: {a[i]} × {b[i]} = {product}")

# %%
dot_product

# %% [markdown]
#
# ### Geometrische Bedeutung
#
# Das Skalarprodukt misst gleichzeitig:
# - Ob zwei Vektoren in ähnliche Richtung zeigen
#   - Größer wenn Vektoren in ähnliche Richtung zeigen
#   - Null wenn Vektoren senkrecht stehen
#   - Negativ wenn Vektoren entgegengesetzt zeigen
# - Wie lang die Vektoren sind
#   - Bei Vektoren in gleicher Richtung: je länger, desto größer das Skalarprodukt

# %%
import matplotlib.pyplot as plt
from linear_regression_plots import plot_dot_product_relationship

# %% [markdown]
#
# Visualisiere Skalarprodukt-Beziehung

# %%

# %% [markdown]
#
# ### Skalarprodukt in NumPy
#
# NumPy macht das Skalarprodukt einfach:

# %%
import numpy as np

# %%
a = np.array([2, 3, 4])
b = np.array([5, 1, 2])

# %% [markdown]
#
# Methode 1: np.dot()

# %%

# %% [markdown]
#
# Methode 2: @ Operator (Python 3.5+)

# %%


# %% [markdown]
#
# Vergleiche mit manueller Berechnung

# %%
a

# %%
b

# %%

# %%

# %% [markdown]
#
# ### Warum ist das wichtig für ML?
#
# Lineare Regression ist ein riesiges Skalarprodukt!
# - Features = ein Vektor
# - Koeffizienten = ein anderer Vektor
# - Vorhersage = Skalarprodukt + Intercept

# %% [markdown]
#
# ### Vorhersagen mit dem Skalarprodukt
#
# Erinnern Sie sich an unser Pizza-Lieferbeispiel:
# - Features: [Entfernung, Verkehr, Anzahl Pizzen]
# - Koeffizienten: [2.0, 1.5, 0.5]
# - Intercept: 8.0

# %%
features = np.array([5.0, 7.0, 3.0])  # distance, traffic, pizzas
coefficients = np.array([2.0, 1.5, 0.5])
intercept = 8.0

# %% [markdown]
#
# Alte Methode mit Schleife

# %%
prediction_loop = 0
for f, c in zip(features, coefficients):
    prediction_loop += f * c
prediction_loop += intercept
print(f"Prediction (loop): {prediction_loop}")

# %% [markdown]
#
# Neue Methode mit Skalarprodukt

# %%

# %% [markdown]
#
# Noch kürzer mit @

# %%

# %% [markdown]
#
# ### Eine Vorhersagefunktion bauen

# %%

# %% [markdown]
#
# ### Testen der Funktion

# %%
test_features = [5.0, 7.0, 3.0]
test_coef = [2.0, 1.5, 0.5]
test_intercept = 8.0

# %%

# %% [markdown]
#
# ### Das Skalarprodukt in der Praxis
#
# Warum ist das Skalarprodukt so wichtig?
# - **Klarheit**: Mathematisch präzise Definition
# - **Universell**: Funktioniert für beliebig viele Features
# - **Optimiert**: NumPy nutzt optimierte C-Bibliotheken

# %% [markdown]
#
# ### Geschwindigkeitsvergleich

# %%
import time

# %% [markdown]
#
# Erstelle große Vektoren

# %%
n = 1_000_000
large_features = np.random.randn(n)
large_coefficients = np.random.randn(n)

# %% [markdown]
#
# Zeitmessung für Schleifenversion

# %%
start = time.time()
result_loop = 0
for f, c in zip(large_features, large_coefficients):
    result_loop += f * c
time_loop = time.time() - start

# %% [markdown]
#
# Zeitmessung für Version mit Comprehension

# %%
start = time.time()
result_comp = sum(f * c for f, c in zip(large_features, large_coefficients))
time_comp = time.time() - start

# %% [markdown]
#
# Zeitmessung für NumPy-Version

# %%
start = time.time()
result_numpy = np.dot(large_features, large_coefficients)
time_numpy = time.time() - start
time_numpy = 1e-8 if time_numpy == 0 else time_numpy  # avoid division by zero on fast systems

# %%
print(f"Loop version: {time_loop:.4f} seconds")
print(f"Comprehension version: {time_comp:.4f} seconds")
print(f"NumPy version: {time_numpy:.4f} seconds")
print(f"NumPy is {time_loop/time_numpy:.1f}x faster than loop!")
print(f"NumPy is {time_comp/time_numpy:.1f}x faster than comprehension!")
print(f"Results match: {np.allclose(result_loop, result_numpy)}")

# %% [markdown]
#
# ## Zusammenfassung
#
# - Das Skalarprodukt ist die Kernoperation der linearen Regression
# - Multipliziere entsprechende Elemente, dann summiere
# - NumPy: `np.dot(a, b)` oder `a @ b`
# - Vorhersage = features @ coefficients + intercept
# - Viel schneller und klarer als Schleifen!

# %% [markdown]
#
# ## Nächste Schritte
#
# - Wir können jetzt einzelne Vorhersagen effizient berechnen
# - Was ist mit vielen Vorhersagen gleichzeitig?
# - Nächstes Thema: Matrizen und Batch-Vorhersagen
# - Von Vektoren zu 2D-Arrays!

# %% [markdown]
#
# ## Workshop: Hauspreisvorhersage mit Skalarprodukt
#
# ### Modell für Hauspreise

# %%
feature_names = ["size_sqm", "bedrooms", "bathrooms", "age_years", "has_garage"]
model_coefficients = np.array([0.5, 25.0, 20.0, -1.5, 30.0])  # in thousands
model_intercept = 100.0  # base price in thousands

# %%
print("House Price Model (in thousands €):")
for name, coef in zip(feature_names, model_coefficients):
    print(f"{name:15}: {coef:+8.1f}k€")
print(f"{'Base price':15}: {model_intercept:8.1f}k€")

# %% [markdown]
#
# ### Aufgabe 1: Einzelne Vorhersage
#
# Berechnen Sie den Preis für dieses Haus mit dem Skalarprodukt:
# - 150 m²
# - 4 Schlafzimmer
# - 2 Badezimmer
# - 5 Jahre alt
# - Hat Garage (1 = ja)

# %%

# %% [markdown]
#
# ### Aufgabe 2: Mehrere Häuser vergleichen
#
# Vergleichen Sie drei verschiedene Häuser und finden Sie das beste Preis-Leistungs-Verhältnis
# (Vorhergesagter Preis pro m²).

# %%
houses = {
    "House A": np.array([100, 3, 1, 10, 0]),  # 100m², 3 bed, 1 bath, 10 years, no garage
    "House B": np.array([180, 4, 3, 2, 1]),   # 180m², 4 bed, 3 bath, 2 years, garage
    "House C": np.array([120, 2, 2, 20, 1]),  # 120m², 2 bed, 2 bath, 20 years, garage
}

# %%

# %% [markdown]
#
# ### Aufgabe 3: Feature-Wichtigkeit analysieren
#
# Welches Feature hat den größten Einfluss auf den Preis?
#
# **Tipp:** Schauen Sie sich die Koeffizienten an, aber bedenken Sie auch die
# Wertebereiche der Features!

# %% [markdown]
#
# Bereiche für Features

# %%
typical_ranges = {
    "size_sqm": (100, 180),
    "bedrooms": (2, 4),
    "bathrooms": (1, 3),
    "age_years": (2, 20),
    "has_garage": (0, 1)
}

# %%
