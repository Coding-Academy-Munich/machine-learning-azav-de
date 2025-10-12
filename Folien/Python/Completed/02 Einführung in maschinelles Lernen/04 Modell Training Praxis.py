# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Modell Training Praxis</b>
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
import seaborn as sns

# %%
plt.scatter(data_x, data_y)
plt.show()

# %%
sns.regplot(x=data_x, y=data_y)
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

# %%
total_abs_error = sum(abs(pred - actual) for pred, actual in zip(train_predictions, train_y))
print(f"Total Absolute Error: {total_abs_error:.2f}")

# %%
total_squared_error = sum((pred - actual) ** 2 for pred, actual in zip(train_predictions, train_y))
print(f"Total Squared Error: {total_squared_error:.2f}")

# %%
mae_manual = total_abs_error / len(train_y)
print(f"Mean Absolute Error: {mae_manual:.2f}")

# %%
mse_manual = total_squared_error / len(train_y)
print(f"Mean Squared Error (manual): {mse_manual:.2f}")

# %% [markdown]
#
# Scikit-Learn bietet viele Metriken zur Bewertung von Modellen an:

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
mae = mean_absolute_error(train_y, train_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error(train_y, train_predictions)
print(f"Mean Squared Error: {mse:.2f}")

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
mae = mean_absolute_error(test_y, predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error(test_y, predictions)
print(f"Mean Squared Error: {mse:.2f}")
