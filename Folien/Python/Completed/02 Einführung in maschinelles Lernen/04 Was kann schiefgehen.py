# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Was kann schiefgehen?</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
from example_data import data_x_reshaped, data_y, train_x, train_y, test_x, test_y
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
data_x_reshaped

# %%
data_y

# %% [markdown]
#
# ## Warum Test/Train-Split?
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
        for i in range(len(X)):
            self.data[X[i][0]] = y[i]
        return self

    def predict(self, X):
        return [self.data.get(sample[0], 0) for sample in X]

# %%
memorize_model = MemorizeModel()

# %%
memorize_model.fit(data_x_reshaped, data_y)

# %%
memorize_predictions = memorize_model.predict(data_x_reshaped)
memorize_predictions

# %%
mean_absolute_error(data_y, memorize_predictions)

# %%
memorize_model = MemorizeModel()

# %%
memorize_model.fit(train_x, train_y)

# %%
memorize_predictions = memorize_model.predict(test_x)
memorize_predictions

# %%
mean_absolute_error(train_y, memorize_model.predict(train_x))

# %%
mean_absolute_error(test_y, memorize_model.predict(test_x))

# %%
