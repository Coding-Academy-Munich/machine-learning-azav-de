# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Eigene Modelle Implementieren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Trainieren eines eigenen Modells
#
# <img src="img/program-vs-model-12.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %%
class LinearRegressionModel:
    """A scikit-learn-style simple linear regression model."""
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        """Trains the model using a grid search.

        Args:
            X (list of lists): Training data features, e.g., [[x1], [x2], ...].
            y (list): Training data targets, e.g., [y1, y2, ...].
        """
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
        """Makes predictions for a list of samples.

        Args:
            X (list of lists): Data features to predict on.

        Returns:
            A list of predictions.
        """
        if self.w is None or self.b is None:
            raise ValueError("Model has not been trained yet.")

        return [self.w * sample[0] + self.b for sample in X]

# %%
# Set up data for custom model training
data_x = [0.5, 1.4, 2.1, 3.5, 4.2, 5.1, 6.3, 7.6, 8.2, 9.0]
data_y = [2.1, 3.5, 5.8, 7.1, 9.3, 11.5, 13.2, 16.0, 17.1, 19.2]

train_x = [[x] for x in data_x[:6]]
train_y = data_y[:6]

test_x = [[x] for x in data_x[6:]]
test_y = data_y[6:]

# %%
custom_model = LinearRegressionModel()

# %%
custom_model.fit(train_x, train_y)

# %%
custom_predictions = custom_model.predict(test_x)
custom_predictions

# %%
for pred, actual in zip(custom_predictions, test_y):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

# %%
# Manual calculation of metrics
def mean_absolute_error_manual(y_true, y_pred):
    return sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)

def mean_squared_error_manual(y_true, y_pred):
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

mae = mean_absolute_error_manual(test_y, custom_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error_manual(test_y, custom_predictions)
print(f"Mean Squared Error: {mse:.2f}")

# %% [markdown]
#
# ## Nochmal Test/Train-Split
#
# - Das folgende Modell merkt sich einfach die Trainingsdaten und gibt sie bei
#   der Vorhersage wieder aus.
# - Es generalisiert nicht.

# %%
class MemorizeModel:
    """A model that simply memorizes the training data."""
    def __init__(self):
        self.data = {}

    def fit(self, X, y):
        """Stores the training data in a dictionary.

        Args:
            X (list of lists): Training data features, e.g., [[x1], [x2], ...].
            y (list): Training data targets, e.g., [y1, y2, ...].
        """
        for i in range(len(X)):
            self.data[X[i][0]] = y[i]
        return self

    def predict(self, X):
        """Makes predictions by looking up the training data.

        Args:
            X (list of lists): Data features to predict on.

        Returns:
            A list of predictions.
        """
        return [self.data.get(sample[0], 0) for sample in X]

# %%
memorize_model = MemorizeModel()

# %%
memorize_model.fit(train_x, train_y)

# %%
memorize_predictions = memorize_model.predict(test_x)
memorize_predictions

# %%
for pred, actual in zip(memorize_predictions, test_y):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

# %%
mae = mean_absolute_error_manual(test_y, memorize_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error_manual(test_y, memorize_predictions)
print(f"Mean Squared Error: {mse:.2f}")
