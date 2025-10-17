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
# ## Unser Muster wiederholen
#
# Erinnern Sie sich an unsere Vorhersageberechnung:
# 1. Feature × Koeffizient für jedes Feature
# 2. Alle Produkte addieren
# 3. Intercept addieren

# %%
import numpy as np

# %% [markdown]
#
# Unser vertrautes Beispiel

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

# %% [markdown]
#
# Summe:

# %%
result

# %% [markdown]
#
# ## Dieses Muster hat einen Namen!
#
# **Skalarprodukt** (Dot Product):
# - Multipliziere entsprechende Elemente
# - Addiere alle Produkte
# - Ergebnis ist eine einzelne Zahl (Skalar)

# %% [markdown]
#
# ## Das Skalarprodukt verstehen
#
# Für zwei Vektoren a und b:
# - a = [a₁, a₂, a₃]
# - b = [b₁, b₂, b₃]
# - a · b = a₁×b₁ + a₂×b₂ + a₃×b₃

# %% [markdown]
#
# ## Beispiel: Schritt für Schritt

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
# ## Geometrische Bedeutung
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
from plot_utils import plot_dot_product_relationship

# %% [markdown]
#
# Visualisiere Skalarprodukt-Beziehung

# %%
plot_dot_product_relationship()

# %% [markdown]
#
# ## Skalarprodukt in NumPy
#
# NumPy macht das Skalarprodukt einfach:

# %%
a = np.array([2, 3, 4])
b = np.array([5, 1, 2])

# %% [markdown]
#
# Methode 1: np.dot()

# %%
dot_result = np.dot(a, b)

# %%
dot_result

# %% [markdown]
#
# Methode 2: @ Operator (Python 3.5+)

# %%
dot_result2 = a @ b

# %%
dot_result2


# %% [markdown]
#
# Vergleiche mit manueller Berechnung

# %%
a

# %%
b

# %%
2*5 + 3*1 + 4*2

# %%
int(a @ b)

# %% [markdown]
#
# ## Warum ist das wichtig für ML?
#
# Lineare Regression ist ein riesiges Skalarprodukt!
# - Features = ein Vektor
# - Koeffizienten = ein anderer Vektor
# - Vorhersage = Skalarprodukt + Intercept

# %% [markdown]
#
# ## Vorhersagen mit dem Skalarprodukt
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
prediction_dot = np.dot(features, coefficients) + intercept
print(f"Prediction (dot product): {prediction_dot}")

# %% [markdown]
#
# Noch kürzer mit @

# %%
prediction_at = features @ coefficients + intercept
print(f"Prediction (@ operator): {prediction_at}")

# %% [markdown]
#
# ## Eine Vorhersagefunktion bauen

# %%
def predict_single(features, coefficients, intercept):
    """
    Make a prediction using linear regression.

    Parameters:
    - features: array of feature values
    - coefficients: array of model coefficients
    - intercept: model intercept

    Returns:
    - prediction: single predicted value
    """
    features = np.array(features)
    coefficients = np.array(coefficients)
    return features @ coefficients + intercept

# %% [markdown]
#
# ### Testen der Funktion

# %%
test_features = [5.0, 7.0, 3.0]
test_coef = [2.0, 1.5, 0.5]
test_intercept = 8.0

# %%
predict_single(test_features, test_coef, test_intercept)

# %% [markdown]
#
# ## Das Skalarprodukt in der Praxis
#
# Warum ist das Skalarprodukt so wichtig?
# - **Effizienz**: Eine Operation statt einer Schleife
# - **Klarheit**: Mathematisch präzise
# - **Universell**: Funktioniert für beliebig viele Features
# - **Optimiert**: NumPy nutzt optimierte C-Bibliotheken

# %% [markdown]
#
# ## Geschwindigkeitsvergleich

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
# Zeitmessung für NumPy-Version

# %%
start = time.time()
result_numpy = np.dot(large_features, large_coefficients)
time_numpy = time.time() - start

# %%
print(f"Loop version: {time_loop:.4f} seconds")
print(f"NumPy version: {time_numpy:.4f} seconds")
print(f"NumPy is {time_loop/time_numpy:.1f}x faster!")
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
house1 = np.array([150, 4, 2, 5, 1])

# %% [markdown]
#
# Verwende Skalarprodukt

# %%
price1 = np.dot(house1, model_coefficients) + model_intercept
print(f"House features: {house1}")
print(f"Predicted price: {price1:.1f}k€")

# %% [markdown]
#
# Zeige Berechnungsdetails

# %%
contributions = house1 * model_coefficients
print("\nDetailed calculation:")
for name, feature, coef, contrib in zip(feature_names, house1, model_coefficients, contributions):
    print(f"{name:15}: {feature:3} × {coef:+6.1f} = {contrib:+7.1f}k€")
print(f"{'Base price':15}:              = {model_intercept:+7.1f}k€")
print(f"{'Total':15}:              = {price1:7.1f}k€")

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
print("House Price Comparison:")
print("-" * 50)

# %%
best_value = float('inf')
best_house = None

# %%
for name, features in houses.items():
    price = features @ model_coefficients + model_intercept
    price_per_sqm = price / features[0]  # features[0] is size_sqm

    print(f"{name}:")
    print(f"  Features: {features}")
    print(f"  Price: {price:.1f}k€")
    print(f"  Price per m²: {price_per_sqm:.3f}k€")

    if price_per_sqm < best_value:
        best_value = price_per_sqm
        best_house = name

# %%
print(f"\nBest value: {best_house} at {best_value:.3f}k€ per m²")

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
print("Feature Impact Analysis:")
print("-" * 60)

# %%
for i, (name, (min_val, max_val)) in enumerate(typical_ranges.items()):
    coef = model_coefficients[i]
    max_impact = coef * (max_val - min_val)
    print(f"{name:15}: Coefficient = {coef:+4.1f}k€")
    print(f"{'':15}  Range: {min_val} to {max_val}")
    print(f"{'':15}  Max impact: {abs(max_impact):.1f}k€")
    print()

# %% [markdown]
#
# - Das Feature mit dem größten absoluten Koeffizienten ist nicht immer am wichtigsten!
# - Man muss den Wertebereich berücksichtigen, den es annehmen kann.

# %% [markdown]
#
# Plot der Einflüsse

# %%
import matplotlib.pyplot as plt

# %%
feature_impacts = []
feature_labels = []
for i, (name, (min_val, max_val)) in enumerate(typical_ranges.items()):
    coef = model_coefficients[i]
    max_impact = coef * (max_val - min_val)
    feature_impacts.append(abs(max_impact))
    feature_labels.append(name)

# %%
plt.figure(figsize=(10, 6))
plt.bar(feature_labels, feature_impacts)
plt.ylabel('Impact on Price (k€)')
plt.title('Feature Impact on House Price')
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.barh(feature_labels, feature_impacts)
plt.xlabel('Impact on Price (k€)')
plt.title('Feature Impact on House Price')
plt.show()

