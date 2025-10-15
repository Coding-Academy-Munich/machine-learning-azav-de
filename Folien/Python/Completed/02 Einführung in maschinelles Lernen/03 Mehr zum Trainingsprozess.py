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
assert len(data_x) == len(data_y)
print(f"We have {len(data_x)} labeled data points.")

# %% [markdown]
#
# ## Visualisierung der Daten

# %%
import matplotlib.pyplot as plt

# %%
plt.scatter(data_x, data_y)
plt.show()

# %% [markdown]
#
# ## Aufteilen der Daten in Trainings- und Testdaten
#
# - Nach dem Training muss das Modell bewertet werden
# - Die Bewertung muss auf anderen Daten basieren als das Training
# - Andernfalls könnte der Lernende nur die Beispiele speichern, die er gesehen hat
# - Uns interessiert aber wie gut das Modell auf neuen Daten funktioniert (generalisiert)

# %%
# Splitting the data: 6 for training, 4 for testing
train_x = [[x] for x in data_x[:6]]
train_y = data_y[:6]
train_x, train_y

# %%
test_x = [[x] for x in data_x[6:]]
test_y = data_y[6:]
test_x, test_y

# %%
print(f"Training samples: {len(train_x)}")
print(f"Test samples: {len(test_x)}")

# %% [markdown]
#
# ## Trainieren des Modells

# %%
from sklearn.linear_model import LinearRegression

# %%
model = LinearRegression()

# %%
try:
    print(model.coef_, model.intercept_)
except AttributeError:
    print("Model is not fitted yet.")

# %%
model.fit(train_x, train_y)

# %%
model.coef_, model.intercept_

# %% [markdown]
#
# ## Bewerten des Modells
#
# Reproduktion der Trainingsdaten:

# %%
train_predictions = model.predict(train_x)
train_predictions

# %% [markdown]
#
# Wie groß sind die Abweichungen?

# %%
for pred, actual in zip(train_predictions, train_y):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

# %% [markdown]
#
# Totaler absoluter Fehler:

# %%
total_absolute_error_train = sum(
    abs(pred - actual) for pred, actual in zip(train_predictions, train_y)
)

# %%
total_absolute_error_train

# %% [markdown]
#
# Totaler quadratischer Fehler:

# %%
total_squared_error_train = sum(
    (pred - actual) ** 2 for pred, actual in zip(train_predictions, train_y)
)

# %%
total_squared_error_train

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
total_absolute_error_few = sum(abs(pred - actual) for pred, actual in few_values)

# %%
total_absolute_error_few

# %%
total_absolute_error_many = sum(abs(pred - actual) for pred, actual in many_values)

# %%
total_absolute_error_many

# %% [markdown]
#
# Besser: Fehler pro Wert
#
# Durchschnittlicher absoluter Fehler (MAE - Mean Absolute Error):

# %%
total_absolute_error_train / len(train_y)

# %% [markdown]
#
# Durchschnittlicher quadratischer Fehler (MSE - Mean Squared Error)

# %%
total_squared_error_train / len(train_y)

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
predictions = model.predict(test_x)
predictions

# %%
for pred, actual in zip(predictions, test_y):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

# %%
mean_absolute_error_test = sum(
    abs(pred - actual) for pred, actual in zip(predictions, test_y)
) / len(test_y)
mean_absolute_error_test

# %%
mean_squared_error_test = sum(
    (pred - actual) ** 2 for pred, actual in zip(predictions, test_y)
) / len(test_y)
mean_squared_error_test

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
pizza_distance[:5]

# %%
pizza_time[:5]

# %% [markdown]
#
# Visualisieren Sie die Daten in einem Scatterplot, um die Beziehung zwischen
# Entfernung und Lieferzeit zu sehen.

# %%
import matplotlib.pyplot as plt

# %%
plt.scatter(pizza_distance, pizza_time)
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (min)")
plt.title("Pizza Delivery Time vs Distance")
plt.show()

# %% [markdown]
#
# Teilen Sie die Daten in Trainings- und Testdaten auf.
#
# - Verwenden Sie ca. 60% der Daten für das Training und ca. 40% für den Test
# - Setzen Sie `random_state=42` für Reproduzierbarkeit
# - Denken Sie daran, dass das Modell erwartet, dass die Features in einer
#   2D-Liste sind (Liste von Listen)

# %%
pizza_x = [[d] for d in pizza_distance]

# %%
pizza_x_train = pizza_x[:9]
pizza_y_train = pizza_time[:9]

pizza_x_test = pizza_x[9:]
pizza_y_test = pizza_time[9:]

# %%
print(f"Training samples: {len(pizza_x_train)}")
print(f"Test samples: {len(pizza_x_test)}")

# %%
pizza_x_train[:3], pizza_y_train[:3]

# %% [markdown]
#
# Trainieren Sie ein lineares Regressionsmodell mit den Trainingsdaten.

# %%
from sklearn.linear_model import LinearRegression

# %%
model = LinearRegression()
model.fit(pizza_x_train, pizza_y_train)

# %%
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# %% [markdown]
#
# Bewerten Sie das Modell mit dem Mean Absolute Error (MAE) und dem Mean Squared
# Error (MSE) sowohl für die Trainings- als auch für die Testdaten.

# %%
train_predictions = model.predict(pizza_x_train)
test_predictions = model.predict(pizza_x_test)

# %%
def mean_absolute_error(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

# %%
def mean_squared_error(y_true, y_pred):
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

# %%
print(f"Training MAE: {mean_absolute_error(pizza_y_train, train_predictions):.2f}")
print(f"Training MSE: {mean_squared_error(pizza_y_train, train_predictions):.2f}")

# %%
print(f"Test MAE: {mean_absolute_error(pizza_y_test, test_predictions):.2f}")
print(f"Test MSE: {mean_squared_error(pizza_y_test, test_predictions):.2f}")

# %% [markdown]
#
# Erstellen Sie einen Plot, der zeigt, wie das Modell die Beziehung zwischen
# Entfernung und Lieferzeit gelernt hat:
#
# - Plotten Sie die Originaldaten als Scatterplot
# - Plotten Sie die Vorhersagen des Modells als Linie

# %%
# Create a range of distances for plotting the model's predictions
distance_range = [[x] for x in range(10)]

# %%
time_predictions = model.predict(distance_range)

# %%
plt.scatter(pizza_distance, pizza_time, label="Actual Data", alpha=0.6)
plt.plot(distance_range, time_predictions, color="red", label="Model Prediction")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (min)")
plt.title("Pizza Delivery Time Model")
plt.legend()
plt.show()

# %% [markdown]
#
# Verwenden Sie das Modell, um die Lieferzeit für neue Entfernungen vorherzusagen:
#
# - 2.8 km
# - 5.2 km
# - 9.0 km

# %%
new_distances = [[2.8], [5.2], [9.0]]
new_predictions = model.predict(new_distances)

# %%
for distance, time in zip(new_distances, new_predictions):
    print(f"Distance: {distance[0]} km → Predicted delivery time: {time:.1f} min")
