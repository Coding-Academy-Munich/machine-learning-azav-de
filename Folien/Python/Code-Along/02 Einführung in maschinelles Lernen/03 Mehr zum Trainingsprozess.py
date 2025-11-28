# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Mehr zum Trainingsprozess</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Daten
#
# - 10 Datensätze mit 1 Merkmal
# - Jedes Merkmal ist eine Zahl zwischen 0.5 und 9.0
# - Zu jedem Datensatz gehört ein Label (2.1 bis 19.2)

# %%
data_x = [0.5, 1.4, 2.1, 3.5, 4.2, 5.1, 6.3, 7.6, 8.2, 9.0]
data_y = [2.1, 3.5, 5.8, 7.1, 9.3, 11.5, 13.2, 16.0, 17.1, 19.2]

# %%

# %% [markdown]
#
# ## Visualisierung der Daten

# %%

# %%

# %% [markdown]
#
# ## Aufteilen der Daten in Trainings- und Testdaten
#
# - Nach dem Training muss das Modell bewertet werden
# - Die Bewertung muss auf anderen Daten basieren als das Training
# - Andernfalls könnte der Lernende nur die Beispiele speichern, die er gesehen hat
# - Uns interessiert aber wie gut das Modell auf neuen Daten funktioniert (generalisiert)

# %%

# %%

# %%

# %% [markdown]
#
# ## Trainieren des Modells

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Bewerten des Modells
#
# Reproduktion der Trainingsdaten:

# %%

# %% [markdown]
#
# Wie groß sind die Abweichungen?

# %%

# %% [markdown]
#
# Totaler absoluter Fehler:

# %%

# %%

# %% [markdown]
#
# Totaler quadratischer Fehler:

# %%

# %%

# %% [markdown]
#
# Problem: Größe des Fehlers hängt von der Anzahl der Werte ab!

# %%
few_values = [(n, n + 1) for n in range(1, 11)]
many_values = [(n, n + 1) for n in range(1, 1001)]

# %% [markdown]
#
# Absoluter Fehler:

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Besser: Fehler pro Wert
#
# Durchschnittlicher absoluter Fehler (MAE - Mean Absolute Error):

# %%

# %% [markdown]
#
# Durchschnittlicher quadratischer Fehler (MSE - Mean Squared Error)

# %%

# %% [markdown]
#
# Die durchschnittliche absolute Fehler sagt, wie weit jeder vorhergesagte Wert
# im Durchschnitt vom echten Wert abweicht:

# %%
total_absolute_error_few / len(few_values)

# %%
total_absolute_error_many / len(many_values)

# %% [markdown]
#
# - Der durchschnittliche quadratische Fehler gibt die durchschnittliche
#   quadratische Abweichung an.
# - Dadurch werden größere Abweichungen stärker gewichtet, kleine Abweichungen
#   weniger stark.

# %%
total_absolute_error_few / len(few_values)

# %%
total_absolute_error_many / len(many_values)

# %%
small_errors = [(n, n + 0.1) for n in range(1, 11)]
large_errors = [(n, n + 10) for n in range(1, 11)]

# %%
sum((pred - actual) ** 2 for pred, actual in small_errors) / len(small_errors)

# %%
sum((pred - actual) ** 2 for pred, actual in large_errors) / len(large_errors)


# %% [markdown]
#
# ## Bewerten des Modells
#
# Wir sehen uns an, wie gut das Modell die Testdaten vorhersagen kann, die
# es noch nicht gesehen hat:

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Workshop: Pizza-Lieferzeit
#
# In diesem Workshop werden wir ein einfaches Machine Learning Modell trainieren,
# um die Lieferzeit einer Pizza basierend auf der Entfernung vom Restaurant
# vorherzusagen.

# %% [markdown]
#
# Die Daten sind bereits vorbereitet:
#
# - `pizza_distance`: Entfernung vom Restaurant in km
# - `pizza_time`: Lieferzeit in Minuten

# %%
pizza_distance = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
]

# %%
pizza_time = [15, 18, 20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 50]

# %%

# %%

# %% [markdown]
#
# Visualisieren Sie die Daten in einem Scatterplot, um die Beziehung zwischen
# Entfernung und Lieferzeit zu sehen.

# %%

# %%

# %% [markdown]
#
# Teilen Sie die Daten in Trainings- und Testdaten auf.
#
# - Verwenden Sie ca. 60% der Daten für das Training und ca. 40% für den Test
# - Denken Sie daran, dass das Modell erwartet, dass die Features in einer
#   2D-Liste sind (Liste von Listen)

# %%

# %% [markdown]
#
# Trainieren Sie ein lineares Regressionsmodell mit den Trainingsdaten.

# %%

# %% [markdown]
#
# Bewerten Sie das Modell mit dem Mean Absolute Error (MAE) und dem Mean Squared
# Error (MSE) sowohl für die Trainings- als auch für die Testdaten.

# %%

# %% [markdown]
#
# Erstellen Sie einen Plot, der zeigt, wie das Modell die Beziehung zwischen
# Entfernung und Lieferzeit gelernt hat:
#
# - Plotten Sie die Originaldaten als Scatterplot
# - Plotten Sie die Vorhersagen des Modells als Linie

# %%

# %%

# %%

# %% [markdown]
#
# Verwenden Sie das Modell, um die Lieferzeit für neue Entfernungen vorherzusagen:
#
# - 2.8 km
# - 5.2 km
# - 9.0 km

# %%
