# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Rechnen mit Vektoren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bereits können
#
# - NumPy Arrays erstellen
# - Arrays inspizieren (`.size`, `.shape`, `.dtype`)
# - Elemente mit Index abrufen
#
# Jetzt lernen wir: **Rechnen mit Arrays!**

# %%
import numpy as np

# %% [markdown]
#
# ## Skalare Operationen
#
# **Skalar** = eine einzelne Zahl
#
# Wir können eine Zahl mit einem ganzen Array verrechnen:
# - Addition
# - Subtraktion
# - Multiplikation
# - Division

# %% [markdown]
#
# ### Beispiel: Preise erhöhen

# %%
prices = np.array([10, 20, 15, 30, 25])

# %%

# %% [markdown]
#
# Alle Preise um 5€ erhöhen:

# %%

# %% [markdown]
#
# ### Prozentuale Erhöhung

# %% [markdown]
#
# 10% Preiserhöhung (multiplizieren mit 1.1):

# %%

# %% [markdown]
#
# 20% Rabatt (multiplizieren mit 0.8):

# %%

# %% [markdown]
#
# ### Weitere Operationen

# %%
values = np.array([10, 20, 30, 40, 50])

# %%

# %% [markdown]
#
# Subtrahieren:

# %%

# %% [markdown]
#
# Dividieren:

# %%

# %% [markdown]
#
# Potenzieren (alle Werte quadrieren):

# %%

# %% [markdown]
#
# ## Vergleich: Listen vs. Arrays
#
# Listen können das nicht!

# %%
prices_list = [10, 20, 15, 30, 25]
prices_array = np.array([10, 20, 15, 30, 25])

# %% [markdown]
#
# ### Mit Listen (funktioniert nicht!)

# %% [markdown]
#
# Das gibt einen Fehler:

# %%
# prices_list * 1.1  # Error!

# %% [markdown]
#
# Wir brauchen eine Schleife oder List Comprehension:

# %%

# %% [markdown]
#
# ### Mit Arrays (einfach!)

# %%

# %% [markdown]
#
# Viel kürzer und schneller!

# %% [markdown]
#
# ## Vektoraddition
#
# Wir können zwei Vektoren addieren:
# - Element für Element (elementweise)
# - Beide Vektoren müssen gleich lang sein

# %% [markdown]
#
# ### Beispiel: Monatliche Verkäufe

# %% [markdown]
#
# Verkäufe von Produkt A (5 Monate):

# %%
product_a = np.array([100, 120, 90, 110, 105])

# %% [markdown]
#
# Verkäufe von Produkt B (5 Monate):

# %%
product_b = np.array([80, 85, 95, 90, 88])

# %%

# %%

# %% [markdown]
#
# Gesamtverkäufe pro Monat:

# %%

# %%

# %% [markdown]
#
# - Monat 1: 100 + 80 = 180
# - Monat 2: 120 + 85 = 205
# - Monat 3: 90 + 95 = 185
# - usw.

# %% [markdown]
#
# ### Andere Vektoroperationen

# %% [markdown]
#
# Subtraktion:

# %%

# %% [markdown]
#
# Multiplikation (elementweise!):

# %%

# %% [markdown]
#
# ## Aggregations-Funktionen
#
# Fassen viele Zahlen zu einer zusammen:

# %%
data = np.array([10, 25, 30, 45, 50])

# %%

# %% [markdown]
#
# ### Summe

# %%

# %% [markdown]
#
# 10 + 25 + 30 + 45 + 50 = 160

# %% [markdown]
#
# ### Durchschnitt (Mittelwert)

# %%

# %% [markdown]
#
# Summe geteilt durch Anzahl: 160 / 5 = 32

# %% [markdown]
#
# ### Minimum und Maximum

# %%

# %%

# %% [markdown]
#
# ### Index von Minimum und Maximum

# %% [markdown]
#
# An welcher Position steht der kleinste Wert?

# %%

# %% [markdown]
#
# An welcher Position steht der größte Wert?

# %%

# %% [markdown]
#
# ## Praktisches Beispiel: Temperaturanalyse

# %% [markdown]
#
# Wöchentliche Temperaturen in Celsius:

# %%
temperatures = np.array([18, 20, 22, 19, 21, 23, 20])

# %%

# %% [markdown]
#
# ### Statistiken berechnen

# %%

# %% [markdown]
#
# ### Umrechnung in Fahrenheit
#
# Formel: F = C × 1.8 + 32

# %%

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Skalare Operationen**: Eine Zahl mit allen Elementen verrechnen
# - **Vektoraddition**: Zwei Vektoren elementweise addieren
# - **Aggregationen**: Viele Zahlen zu einer zusammenfassen
#   - `.sum()`, `.mean()`, `.min()`, `.max()`
#   - `.argmin()`, `.argmax()` für Indizes

# %% [markdown]
#
# ## Workshop: Rechnen mit Vektoren
#
# Probieren Sie es selbst!

# %% [markdown]
#
# ### Aufgabe 1: Gehälter anpassen
#
# Ein Unternehmen hat 5 Mitarbeiter mit folgenden Monatsgehältern (in €):

# %%
salaries = np.array([2500, 3200, 2800, 3500, 2900])

# %% [markdown]
#
# 1. Alle bekommen eine Gehaltserhöhung von 200€. Wie hoch sind die neuen Gehälter?
# 2. Das Unternehmen plant eine 5% Gehaltserhöhung. Wie hoch wären die Gehälter dann?
# 3. Was ist das durchschnittliche Gehalt vor und nach der 5% Erhöhung?

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 2: Wöchentliche Schritte
#
# Sie haben Ihre Schritte für zwei Wochen aufgezeichnet:

# %%
week1_steps = np.array([8000, 10000, 7500, 9000, 8500, 12000, 6000])
week2_steps = np.array([7000, 9500, 8000, 10000, 9000, 11000, 7500])

# %% [markdown]
#
# 1. Was ist die Gesamtzahl der Schritte in Woche 1?
# 2. Was ist der Durchschnitt pro Tag in jeder Woche?
# 3. An welchem Wochentag (Index) hatten Sie die meisten Schritte in Woche 1?
# 4. Wie viele Schritte haben Sie insgesamt in beiden Wochen gemacht?

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 3: Online-Shop Umsatz
#
# Ein Online-Shop hat die Verkaufszahlen für 6 Produkte:

# %% [markdown]
#
# Anzahl verkaufter Einheiten pro Produkt:

# %%
units_sold = np.array([45, 120, 78, 34, 90, 156])

# %% [markdown]
#
# Preis pro Einheit (in €):

# %%
unit_prices = np.array([29.99, 15.50, 45.00, 99.99, 12.50, 8.99])

# %% [markdown]
#
# 1. Berechnen Sie den Umsatz pro Produkt (Anzahl × Preis)
# 2. Was ist der Gesamtumsatz?
# 3. Welches Produkt (Index) hat den höchsten Umsatz generiert?
# 4. Was ist der durchschnittliche Umsatz pro Produkt?

# %%
# Your solution here

# %% [markdown]
#
# ### Aufgabe 4: Messwerte normalisieren
#
# Sie haben Messwerte von einem Sensor:

# %%
measurements = np.array([23.5, 45.8, 12.3, 67.2, 34.1, 56.7, 89.0])

# %% [markdown]
#
# Normalisieren Sie die Werte auf den Bereich 0-100:
# 1. Finden Sie den Minimalwert
# 2. Finden Sie den Maximalwert
# 3. Berechnen Sie: `(measurements - min) / (max - min) * 100`

# %%
# Your solution here

# %%
