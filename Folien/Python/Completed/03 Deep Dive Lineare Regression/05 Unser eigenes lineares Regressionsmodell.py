# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Unser eigenes lineares Regressionsmodell</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was macht ein Modell aus?
#
# Ein lineares Regressionsmodell braucht:
# - **Koeffizienten** (Gewichte für Features)
# - **Intercept** (Basiswert)
# - **predict()** Methode für Vorhersagen
# - **fit()** Methode zum Lernen (kommt später)

# %%
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# %% [markdown]
#
# ## Scikit-Learn's Modellstruktur
#
# Wir trainieren ein einfaches Scikit-Learn-Modell

# %%
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = np.array([5, 7, 9, 11, 13])

# %%
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

# %%
print(f"Coefficients: {sklearn_model.coef_}")
print(f"Intercept: {sklearn_model.intercept_}")

# %% [markdown]
#
# Vorhersagen mit dem trainierten Modell

# %%
X_test = np.array([[6, 7], [7, 8]])

# %%
sklearn_predictions = sklearn_model.predict(X_test)

# %%
sklearn_predictions

# %% [markdown]
#
# ## Unsere erste Modellklasse
#
# Wir wollen eine eigene Klasse implementieren, die ähnlich funktioniert

# %%
class SimpleLinearRegression:
    """A simple linear regression model."""

    def __init__(self, coefficients=None, intercept=0.0):
        """
        Initialize the model.

        Parameters:
        - coefficients: array of feature weights
        - intercept: base value (bias term)
        """
        self.coef_ = np.array(coefficients) if coefficients is not None else None
        self.intercept_ = intercept

    def __repr__(self):
        return f"SimpleLinearRegression(coef={self.coef_}, intercept={self.intercept_})"

# %% [markdown]
#
# ## Modell mit bekannten Parametern erstellen
#
# Erstelle das Modell mit Scikit-Learns gelernten Parametern

# %%
our_model = SimpleLinearRegression(
    coefficients=sklearn_model.coef_,
    intercept=sklearn_model.intercept_
)

# %%
print(our_model)

# %% [markdown]
#
# ## Die predict() Methode implementieren

# %%
class SimpleLinearRegression:
    """A simple linear regression model."""

    def __init__(self, coefficients=None, intercept=0.0):
        self.coef_ = np.array(coefficients) if coefficients is not None else None
        self.intercept_ = intercept

    def predict(self, X):
        """
        Make predictions for input data.

        Parameters:
        - X: 2D array of shape (n_samples, n_features)

        Returns:
        - predictions: 1D array of shape (n_samples,)
        """
        X = np.array(X)

        # Check if model has coefficients
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet!")

        # Handle 1D input (single sample)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Check dimensions
        if X.shape[1] != len(self.coef_):
            raise ValueError(
                f"Feature count mismatch: input has {X.shape[1]} features, "
                f"model expects {len(self.coef_)}"
            )

        # Make predictions using matrix multiplication
        predictions = X @ self.coef_ + self.intercept_

        return predictions

# %% [markdown]
#
# ## Testen unseres Modells
#
# - Erstelle Modell mit bekannten Parametern
# - Teste Vorhersagen

# %%
our_model = SimpleLinearRegression(
    coefficients=sklearn_model.coef_,
    intercept=sklearn_model.intercept_
)

# %% [markdown]
#
# Teste Vorhersagen

# %%
our_predictions = our_model.predict(X_test)

# %%
our_predictions

# %%
sklearn_predictions

# %% [markdown]
#
# ## Einzelne Vorhersage
#
# Teste Vorhersage für ein einzelnes Sample

# %%
single_sample = [8, 9]

# %%
single_pred = our_model.predict(single_sample)

# %%
single_pred

# %%
sklearn_model.predict([single_sample])

# %% [markdown]
#
# ## Modell mit Zusatzfunktionen

# %%
class SimpleLinearRegression:
    """A simple linear regression model with additional features."""

    def __init__(self, coefficients=None, intercept=0.0):
        self.coef_ = np.array(coefficients) if coefficients is not None else None
        self.intercept_ = intercept
        self.n_features_ = len(self.coef_) if self.coef_ is not None else None

    def predict(self, X):
        """Make predictions for input data."""
        X = np.array(X)

        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet!")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self.coef_):
            raise ValueError(
                f"Feature count mismatch: input has {X.shape[1]} features, "
                f"model expects {len(self.coef_)}"
            )

        return X @ self.coef_ + self.intercept_

    def get_params(self):
        """Get model parameters."""
        return {
            'coefficients': self.coef_,
            'intercept': self.intercept_,
            'n_features': self.n_features_
        }

    def score_sample(self, x):
        """
        Score a single sample (show contribution of each feature).

        Parameters:
        - x: 1D array of features

        Returns:
        - dict with feature contributions
        """
        x = np.array(x)
        if x.ndim != 1:
            raise ValueError("Expected 1D array for single sample")

        if len(x) != len(self.coef_):
            raise ValueError(f"Expected {len(self.coef_)} features, got {len(x)}")

        contributions = x * self.coef_

        return {
            'features': x,
            'coefficients': self.coef_,
            'contributions': contributions,
            'intercept': self.intercept_,
            'total': contributions.sum() + self.intercept_
        }

# %% [markdown]
#
# ## Feature-Beiträge analysieren
#
# - Erstelle Modell
# - Analysiere ein Sample

# %%
model = SimpleLinearRegression(
    coefficients=[2.0, 1.5, -0.5],
    intercept=10.0
)

# %%
sample = [5, 3, 2]
analysis = model.score_sample(sample)

# %%
analysis

# %%
print("Feature Analysis:")
print("-" * 40)
for i in range(len(sample)):
    print(f"Feature {i}: {analysis['features'][i]:6.1f} × {analysis['coefficients'][i]:+6.2f} = {analysis['contributions'][i]:+7.2f}")
print(f"Intercept:                 = {analysis['intercept']:+7.2f}")
print("-" * 40)
print(f"Total prediction:          = {analysis['total']:7.2f}")


# %% [markdown]
#
# ## Zusammenfassung
#
# - Wir haben unser eigenes Regressionsmodell gebaut!
# - Kernkomponenten: Koeffizienten, Intercept, predict()
# - Nutzt Matrix-Multiplikation für effiziente Vorhersagen
# - Kann erweitert werden mit Analyse- und Visualisierungsmethoden
# - Verhält sich wie Scikit-Learn's LinearRegression

# %% [markdown]
#
# ## Nächste Schritte
#
# - Unser Modell kann vorhersagen, aber noch nicht lernen
# - Wie findet man die optimalen Koeffizienten?
# - Nächstes Thema: Die fit() Methode implementieren
# - Lernen aus Daten mit der Normalgleichung!

# %% [markdown]
#
# ## Workshop: Erweiterte Modellklasse

# %% [markdown]
#
# Autopreismodell für Workshop

# %%
car_features = ["age", "mileage", "engine_size", "num_owners"]
car_coefficients = [-2.5, -0.08, 12.0, -3.0]  # in thousands
car_intercept = 35.0

# %% [markdown]
#
# Testdaten

# %%
cars = np.array([
    [3, 45, 2.0, 1],   # 3 years, 45k km, 2.0L, 1 owner
    [5, 80, 1.6, 2],   # 5 years, 80k km, 1.6L, 2 owners
    [1, 10, 3.0, 0],   # 1 year, 10k km, 3.0L, new
    [8, 120, 1.4, 3],  # 8 years, 120k km, 1.4L, 3 owners
])

# %% [markdown]
#
# ### Aufgabe 1: Modell erstellen und testen
#
# Erstellen Sie ein Modell für Autopreise und machen Sie Vorhersagen:

# %% [markdown]
#
# Erstelle Autopreismodell

# %%
car_model = SimpleLinearRegression(
    coefficients=car_coefficients,
    intercept=car_intercept
)

# %% [markdown]
#
# Mache Vorhersagen

# %%
car_prices = car_model.predict(cars)

# %%
print("Car Price Predictions:")
print("-" * 40)
for i, (car_data, price) in enumerate(zip(cars, car_prices)):
    print(f"Car {i+1}: {price:6.1f}k€")
    print(f"  Features: {car_data}")

# %% [markdown]
#
# ### Aufgabe 2: Detailanalyse
#
# Analysieren Sie, welches Feature den größten Einfluss auf den Preis von Auto 3 hat:

# %% [markdown]
#
# Analysiere Auto 3

# %%
car3_analysis = car_model.score_sample(cars[2])

# %%
print("Car 3 - Detailed Price Breakdown:")
print("-" * 50)
for i, feature_name in enumerate(car_features):
    feature_val = car3_analysis['features'][i]
    coef = car3_analysis['coefficients'][i]
    contrib = car3_analysis['contributions'][i]
    print(f"{feature_name:12}: {feature_val:6.1f} × {coef:+6.2f} = {contrib:+7.2f}k€")

print(f"{'Base price':12}:                 = {car3_analysis['intercept']:+7.2f}k€")
print("-" * 50)
print(f"{'Total':12}:                 = {car3_analysis['total']:7.2f}k€")

# %% [markdown]
#
# Finde größten Einfluss

# %%
abs_contributions = np.abs(car3_analysis['contributions'])
biggest_impact_idx = abs_contributions.argmax()
print(f"\nBiggest impact: {car_features[biggest_impact_idx]} ({car3_analysis['contributions'][biggest_impact_idx]:+.1f}k€)")
