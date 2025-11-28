# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Multivariate Vorhersagen verstehen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bereits wissen
#
# - Univariate Regression: Eine Eingabe → Eine Ausgabe
# - Beispiel: `Lieferzeit = 2 × Entfernung + 10`
# - Wir haben bereits multivariate Modelle mit Scikit-Learn trainiert
# - Aber wie funktionieren sie wirklich?

# %% [markdown]
#
# ## Erinnerung: Univariate Regression
#
# Pizza-Lieferzeit basierend auf Entfernung: Einfaches univariates Modell

# %%
distance = 5.0  # km
coefficient = 2.0  # minutes per km
intercept = 10.0  # base time in minutes

# %%
delivery_time = coefficient * distance + intercept

# %% [markdown]
#
# Entfernung ist das einzige Feature:

# %%
distance

# %% [markdown]
#
# Als Ergebnis erhalten wir die vorhergesagte Lieferzeit:

# %%
delivery_time

# %% [markdown]
#
# ## Von einem zu mehreren Features
#
# Was beeinflusst die Lieferzeit noch?
# - Entfernung (km)
# - Verkehrslage (0-10 Skala)
# - Anzahl der Pizzen
#
# Wir haben jetzt mehrere Features!

# %%
distance = 5.0
traffic = 7.0  # heavy traffic
num_pizzas = 3

# %% [markdown]
#
# Wie kombinieren wir diese?

# %% [markdown]
#
# ## Multivariate Vorhersage: Die Idee
#
# Jedes Feature trägt zur Vorhersage bei:
# - Jedes Feature hat einen eigenen Koeffizienten
# - Wir multiplizieren Feature × Koeffizient
# - Dann addieren wir alle Beiträge

# %% [markdown]
#
# ## Beispiel: Pizza-Lieferzeit mit 3 Features

# %% [markdown]
#
# Gegebene Modellkoeffizienten (bereits trainiert)

# %%
coef_distance = 2.0    # minutes per km
coef_traffic = 1.5     # minutes per traffic unit
coef_pizzas = 0.5      # minutes per pizza
intercept = 8.0        # base time

# %% [markdown]
#
# Features für eine Bestellung

# %%
distance = 5.0
traffic = 7.0
num_pizzas = 3

# %% [markdown]
#
# Berechne Vorhersage Schritt für Schritt

# %%
contribution_distance = coef_distance * distance
contribution_traffic = coef_traffic * traffic
contribution_pizzas = coef_pizzas * num_pizzas

# %%
print(f"Distance contribution: {coef_distance} × {distance} = {contribution_distance} min")
print(f"Traffic contribution: {coef_traffic} × {traffic} = {contribution_traffic} min")
print(f"Pizza contribution: {coef_pizzas} × {num_pizzas} = {contribution_pizzas} min")
print(f"Base time (intercept): {intercept} min")

# %%
total_time = contribution_distance + contribution_traffic + contribution_pizzas + intercept

# %%
print(f"\nTotal predicted time: {total_time} minutes")

# %% [markdown]
#
# ## Das Muster erkennen
#
# Die Berechnung folgt immer dem gleichen Muster:
# 1. Multipliziere jedes Feature mit seinem Koeffizienten
# 2. Addiere alle Produkte
# 3. Addiere den Intercept

# %% [markdown]
#
# ## Mit Listen organisieren
#
# - Organisiere Features und Koeffizienten als Listen

# %%
features = [distance, traffic, num_pizzas]
coefficients = [coef_distance, coef_traffic, coef_pizzas]  # corresponding coefficients
intercept = 8.0

# %%
feature_names = ["distance_km", "traffic_level", "num_pizzas"]

# %% [markdown]
#
# Berechne mit Listen

# %%
prediction = 0
for i in range(len(features)):
    contribution = features[i] * coefficients[i]
    prediction += contribution
    print(f"Feature {i}: {features[i]} × {coefficients[i]} = {contribution}")

# %%
prediction += intercept
print(f"Plus intercept: {intercept}")
print(f"Total prediction: {prediction}")

# %% [markdown]
#
# ## Kompaktere Schreibweise
#
# - Verwende zip um Features mit Koeffizienten zu paaren

# %%
prediction = sum(f * c for f, c in zip(features, coefficients)) + intercept

# %%
print(f"Prediction: {prediction} minutes")

# %% [markdown]
#
# Welches Feature hat den größten Einfluss?

# %%
contributions = [f * c for f, c in zip(features, coefficients)]

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Wie können wir den Einfluss der Features visualisieren?

# %%
import matplotlib.pyplot as plt

# %%
plt.bar(range(len(features)), contributions)
plt.xticks(range(len(features)), [f"{name}" for name in feature_names])
plt.ylabel("Contribution to Prediction")
plt.title("Feature Contributions to Prediction")
plt.show()

# %% [markdown]
#
# ## Beispiel: Hauspreise
#
# Ein komplexeres Beispiel mit mehr Features:
# Hauspreismodell (Koeffizienten in Tausend Euro)

# %%
feature_names = ["size_sqm", "bedrooms", "bathrooms", "age_years", "garage"]
coefficients = [0.3, 20.0, 15.0, -2.0, 25.0]
intercept = 50.0  # base price in thousands

# %%
print("House Price Model:")
print("-" * 40)
for name, coef in zip(feature_names, coefficients):
    print(f"{name:15} : {coef:+8.1f}k€")
print(f"{'Base price':15} : {intercept:8.1f}k€")

# %% [markdown]
#
# ### Vorhersage für ein konkretes Haus

# %%
house_features = [120, 3, 2, 10, 1]  # 120sqm, 3 bed, 2 bath, 10 years old, 1 garage

# %%
print("House characteristics:")
for name, value in zip(feature_names, house_features):
    print(f"{name:15} : {value}")

# %% [markdown]
#
# Berechne Preis

# %%
price = 0
print("\nPrice calculation:")
for name, feature, coef in zip(feature_names, house_features, coefficients):
    contribution = feature * coef
    price += contribution
    print(f"{name:15} : {feature:3} × {coef:+6.1f} = {contribution:+7.1f}k€")

# %%
price += intercept
print(f"{'Base price':15} :              = {intercept:+7.1f}k€")
print("-" * 45)
print(f"{'Total price':15} :              = {price:7.1f}k€")

# %% [markdown]
#
# ## Zusammenfassung
#
# - Multivariate Regression: Mehrere Features → Eine Vorhersage
# - Jedes Feature hat einen Koeffizienten (Gewicht)
# - Vorhersage = Summe(Feature × Koeffizient) + Intercept
# - Das ist die Grundlage für komplexere Modelle!

# %% [markdown]
#
# ## Nächste Schritte
#
# - Diese Berechnungen werden schnell mühsam
# - Mit vielen Features und Datenpunkten brauchen wir bessere Tools
# - NumPy macht diese Berechnungen einfach und schnell
# - Als Nächstes: Einführung in NumPy und Vektoren

# %% [markdown]
#
# ## Workshop: Autopreise vorhersagen
#
# Sie arbeiten für einen Gebrauchtwagenhändler und haben ein Modell
# zur Preisvorhersage mit folgenden Features:

# %% [markdown]
#
# ### Autopreismodell

# %%
car_features = ["age_years", "mileage_1000km", "engine_liters", "num_owners", "is_diesel"]
car_coefficients = [-2.5, -0.15, 8.0, -1.5, 3.0]  # in thousands of euros
car_intercept = 25.0  # base price in thousands

# %%
print("Car Price Model (coefficients in k€):")
for feature, coef in zip(car_features, car_coefficients):
    print(f"{feature:20} : {coef:+6.2f}")
print(f"{'Base price':20} : {car_intercept:6.2f}")

# %% [markdown]
#
# ### Aufgabe 1: Einzelne Vorhersage
#
# Berechnen Sie den Preis für folgendes Auto:
# - 3 Jahre alt
# - 45.000 km (= 45 in Tausend km)
# - 2.0 Liter Motor
# - 1 Vorbesitzer
# - Diesel (1 = ja, 0 = nein)

# %%

# %% [markdown]
#
# ### Aufgabe 2: Mehrere Vorhersagen
#
# Berechnen Sie die Preise für diese drei Autos:

# %%
cars = [
    [5, 80, 1.6, 2, 0],  # Car A: 5 years, 80k km, 1.6L, 2 owners, petrol
    [1, 10, 3.0, 0, 1],  # Car B: 1 year, 10k km, 3.0L, 0 owners (new), diesel
    [8, 120, 1.4, 3, 0], # Car C: 8 years, 120k km, 1.4L, 3 owners, petrol
]

# %%
car_names = ["Car A", "Car B", "Car C"]

# %%

# %% [markdown]
#
# ### Aufgabe 3: Feature-Analyse
#
# Welches Feature hat den größten Einfluss auf den Preis?
#
# Berechnen Sie für Car B, welches Feature am meisten zum Preis beiträgt.

# %%

# %%
