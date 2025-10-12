# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Overfitting und Generalisierung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Overfitting
#
# - Ein Modell, das die Trainingsdaten perfekt vorhersagt, aber bei neuen Daten
#   versagt, hat überangepasst (overfitted).
# - Es hat die Trainingsdaten auswendig gelernt, anstatt die zugrunde liegenden
#   Muster zu erkennen.
# - Overfitting ist ein häufiges Problem im maschinellen Lernen.

# %% [markdown]
#
# ## Verbesserung des Memorize-Modells
#
# - Anstatt nur exakte Übereinstimmungen zurckzugeben, geben wir den Wert des
#   nächsten Trainingspunkts zurück.
# - Dies ist eine einfache Form von sogenanntem k-Nearest Neighbors (k-NN).

# %%
import math

# %%
class NearestNeighborModel:
    """A scikit-learn-style 1-Nearest Neighbor model."""
    def __init__(self):
        self.train_X = None
        self.train_y = None

    def fit(self, X, y):
        """Stores the training data."""
        self.train_X = X
        self.train_y = y
        return self

    def predict(self, X):
        """Predicts for each sample in X by finding its nearest neighbor."""
        predictions = []
        for sample_to_predict in X:
            best_dist = float('inf')
            best_y = None
            for i in range(len(self.train_X)):
                dist = abs(self.train_X[i][0] - sample_to_predict[0])
                if dist < best_dist:
                    best_dist = dist
                    best_y = self.train_y[i]
            predictions.append(best_y)
        return predictions

# %%
# Set up data
data_x = [0.5, 1.4, 2.1, 3.5, 4.2, 5.1, 6.3, 7.6, 8.2, 9.0]
data_y = [2.1, 3.5, 5.8, 7.1, 9.3, 11.5, 13.2, 16.0, 17.1, 19.2]

train_x = [[x] for x in data_x[:6]]
train_y = data_y[:6]

test_x = [[x] for x in data_x[6:]]
test_y = data_y[6:]

# %%
nn_model = NearestNeighborModel()

# %%
nn_model.fit(train_x, train_y)

# %%
nn_predictions = nn_model.predict(test_x)
nn_predictions

# %%
def mean_absolute_error_manual(y_true, y_pred):
    return sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)

def mean_squared_error_manual(y_true, y_pred):
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

mae = mean_absolute_error_manual(test_y, nn_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error_manual(test_y, nn_predictions)
print(f"Mean Squared Error: {mse:.2f}")

# %% [markdown]
#
# Warum ist das NN-Modell immer noch so schlecht?

# %%
for pred, actual in zip(nn_predictions, test_y):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

# %% [markdown]
#
# ## Randomisierung des Datensatzes
#
# - Die ursprünglichen Daten waren sortiert.
# - Dadurch waren die Trainings- und Testdaten sehr unterschiedlich.
# - Das Problem ist damit nicht, Werte zwischen den Trainingspunkten
#   vorherzusagen, sondern aus dem Bereich der Trainingsdaten heraus zu
#   extrapolieren.
# - Durch Mischen der Daten wird dieses Problem behoben.

# %%
import random

# %%
combined = list(zip(data_x, data_y))
random.shuffle(combined)
data_x[:], data_y[:] = zip(*combined)

# %%
data_x, data_y

# %%
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

# %%
nn_model = NearestNeighborModel()
nn_model.fit(train_x, train_y)
nn_predictions = nn_model.predict(test_x)
nn_predictions

# %%
mae = mean_absolute_error_manual(test_y, nn_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error_manual(test_y, nn_predictions)
print(f"Mean Squared Error: {mse:.2f}")

# %%
for pred, actual in zip(nn_predictions, test_y):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

# %%
# Custom Linear Regression Model with grid search
class LinearRegressionModel:
    """A scikit-learn-style simple linear regression model."""
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        """Trains the model using a grid search."""
        best_w, best_b, min_loss = 0, 0, float('inf')
        possible_ws = [1.5, 1.8, 2.0, 2.2]
        possible_bs = [0.5, 1.0, 1.5]

        for w_guess in possible_ws:
            for b_guess in possible_bs:
                predictions = [w_guess * sample[0] + b_guess for sample in X]
                errors = [y[i] - predictions[i] for i in range(len(y))]
                mse = sum(e**2 for e in errors) / len(y)

                if mse < min_loss:
                    min_loss = mse
                    best_w, best_b = w_guess, b_guess

        self.w, self.b = best_w, best_b
        print(f"Learned parameters: w={self.w}, b={self.b}")
        return self

    def predict(self, X):
        """Makes predictions for a list of samples."""
        if self.w is None or self.b is None:
            raise ValueError("Model has not been trained yet.")
        return [self.w * sample[0] + self.b for sample in X]

# %%
custom_model = LinearRegressionModel()
custom_model.fit(train_x, train_y)
custom_predictions = custom_model.predict(test_x)
custom_predictions

# %%
mae = mean_absolute_error_manual(test_y, custom_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error_manual(test_y, custom_predictions)
print(f"Mean Squared Error: {mse:.2f}")
