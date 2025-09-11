# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Was ist Maschinelles Lernen?</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# <img src="img/robots-playing-checkers.png"
#      style="width: 90%; margin-left: auto; margin-right: auto; display: block;">

# %% [markdown]
#
# ## Die Ursprünge des maschinellen Lernens
#
# - Ein Computer kann so programmiert werden, dass er lernt, besser Dame zu spielen,
#   als der Programmierer, der das Programm geschrieben hat.<br>
#   *(Arthur Samuel, 1959)*
#
# - Das Programmieren von Computern, um aus Erfahrung zu lernen, sollte letztendlich
#   die Notwendigkeit für einen Großteil dieses detaillierten Programmieraufwands
#   beseitigen.<br>
#   *(Arthur Samuel, 1959)*

# %% [markdown]
#
# <div style="text-align: center; font-size: 200%; line-height: 150%;">
#   Was ist Maschinelles Lernen?
# </div>

# %% [markdown]
#
# ## Eine mögliche Antwort (Andrew Glassner)
#
# Der Begriff maschinelles Lernen beschreibt eine wachsende Zahl von Techniken, die
# alle ein Ziel haben: aussagekräftige Informationen aus Daten zu gewinnen.
#
# „Daten“ bezieht sich hier auf alles, was aufgezeichnet und gemessen werden kann.
# [...]
#
# "Sinnvolle Information" ist alles, was wir aus den Daten extrahieren können, was
# uns in irgendeiner Weise nützlich sein wird.

# %% [markdown]
#
# ## Andrew Glassners Bücher
#
# <img src="img/glassner-book.jpg"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Sinnvolle Information aus Daten (1)
#
# <img src="img/Figure-01-001.png">
#
# Postleitzahlen erkennen, Schecks validieren, Gesichter erkennen

# %% [markdown]
#
# ## Sinnvolle Information aus Daten (2)
#
# <img src="img/Figure-01-002.png">
#
# Sprache verstehen, seltene Ereignisse erkennen, Vorhersagen treffen

# %% [markdown]
#
# ## Eine andere Antwort (nach von François Chollet)
#
# - Ein Teil der Künstlichen Intelligenz (KI)
# - KI: Computer dazu bringen, Probleme zu lösen, die zuvor nur von Menschen gelöst
#   werden konnten
# - KI muss nicht lernen, z. B. Expertensysteme
# - ML: Verhalten mit zusätzlichen Daten verbessern

# %% [markdown]
#
# ## Biologische Inspiration
#
# <img src="img/Figure-10-001.png"
#      style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-01.png"
#      style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-02.png"
#      style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ### Installation der benötigten Bibliotheken

# %%
# #!pip install matplotlib seaborn scikit-learn pandas numpy transformers huggingface_hub[hf_xet]

# %% [markdown]
#
# Installation von PyTorch mit CUDA-Unterstützung (falls GPU vorhanden)

# %%
# #!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# %% [markdown]
#
# Installation von PyTorch ohne CUDA-Unterstützung (CPU only)

# %%
# #!pip install torch torchvision

# %% [markdown]
#
# ## Ausführen eines Programms

# %%
def add(a, b):
    return a + b

# %%
add(2, 3)

# %% [markdown]
#
# ## Ausführen eines Modells

# %%
from transformers import pipeline

# %%
sentiment_analysis_model = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# %%
sentiment_analysis_model("I love machine learning!")

# %%
sentiment_analysis_model("I hate bad weather!")

# %%
sentiment_analysis_model("It's all right, I guess.")

# %% [markdown]
#
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-03.png"
#      style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-04.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-06.png"
#      style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-03.png"
#      style="width: 30%; margin-left: auto; margin-right: auto;"/>
# <img src="img/program-vs-model-06.png"
#      style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Beispiel: MNIST-Daten
#
# <img src="img/Figure-01-023.png" style="float: right;width: 40%;"/>
#
# Wir wollen handgeschriebene Ziffern erkennen:

# %% [markdown]
#
# ## Regelbasierte Systeme: Feature Engineering
#
# Extraktion relevanter Merkmale aus Daten durch Menschen.

# %% [markdown]
#
# <img src="img/Figure-01-003.png"
#      style="width: 40%; margin-left: auto; margin-right: auto;">

# %% [markdown]
#
# <img src="img/Figure-01-004.png"
#      style="width: 20%; margin-left: auto; margin-right: auto;">

# %% [markdown]
#
# <div style="text-align: center; font-size: 200%; line-height: 150%;">
#   Überwachtes Lernen
# </div>

# %% [markdown]
#
# ## Gelabelte Daten
#
# <img src="img/Figure-01-007.png"
#      style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Trainieren eines Modells mit überwachtem Lernen
#
# - Zeige einem Lern-Algorithmus viele mit Labels versehene Daten
# - Prüfe, wie gut Labels anhand der Merkmale korrekt zuordnen kann
# - Korrigiere die Parameter des Modells, wenn er falsch liegt

# %% [markdown]
#
# ## Trainieren eines vordefinierten Modells
#
# <img src="img/program-vs-model-10.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

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
mean_abs_error = total_abs_error / len(train_y)
print(f"Mean Absolute Error: {mean_abs_error:.2f}")

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
mae = mean_absolute_error(test_y, custom_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error(test_y, custom_predictions)
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
mae = mean_absolute_error(test_y, memorize_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error(test_y, memorize_predictions)
print(f"Mean Squared Error: {mse:.2f}")

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
nn_model = NearestNeighborModel()

# %%
nn_model.fit(train_x, train_y)

# %%
nn_predictions = nn_model.predict(test_x)
nn_predictions

# %%
mae = mean_absolute_error(test_y, nn_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error(test_y, nn_predictions)
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
mae = mean_absolute_error(test_y, nn_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error(test_y, nn_predictions)
print(f"Mean Squared Error: {mse:.2f}")

# %%
for pred, actual in zip(nn_predictions, test_y):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

# %%
custom_model = LinearRegressionModel()
custom_model.fit(train_x, train_y)
custom_predictions = custom_model.predict(test_x)
custom_predictions

# %%
mae = mean_absolute_error(test_y, custom_predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# %%
mse = mean_squared_error(test_y, custom_predictions)
print(f"Mean Squared Error: {mse:.2f}")


# %% [markdown]
#
# ## Der ML Workflow
#
# <img src="img/ml-workflow-data.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Der ML Workflow
#
# <img src="img/ml-workflow-backtrack.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Auswahl eines Modells
#
# <img src="img/choosing-ml-models-03.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Auswahl eines Modells
#
# <img src="img/choosing-ml-models-05.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Auswahl eines Modells
#
# <img src="img/choosing-ml-models-07.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Auswahl eines Modells
#
# <img src="img/choosing-ml-models-09.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>
