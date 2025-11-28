# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Machine Learning mit Scikit-Learn</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was ist Scikit-Learn?
#
# - Scikit-Learn ist eine der beliebtesten Python-Bibliotheken für Machine Learning
# - Bietet einfache und effiziente Tools für Datenanalyse und -modellierung
# - Basiert auf NumPy, SciPy und Matplotlib
# - Open Source und kommerziell nutzbar (BSD-Lizenz)

# %% [markdown]
#
# ## Die wichtigsten Module in Scikit-Learn
#
# - **Supervised Learning**: Algorithmen für klassifizierte/beschriftete Daten
#   - `sklearn.linear_model`: Lineare Modelle (Regression, Ridge, Lasso, etc.)
#   - `sklearn.tree`: Entscheidungsbäume
#   - `sklearn.ensemble`: Ensemble-Methoden (Random Forests, Boosting)
#   - `sklearn.svm`: Support Vector Machines
#   - `sklearn.neighbors`: Nearest Neighbors
#   - `sklearn.naive_bayes`: Naive Bayes Klassifikatoren

# %% [markdown]
#
# ## Weitere wichtige Module
#
# - **Unsupervised Learning**: Algorithmen für unbeschriftete Daten
#   - `sklearn.cluster`: Clustering-Algorithmen
#   - `sklearn.decomposition`: Dimensionsreduktion
# - **Model Selection**: Tools für Modellauswahl und -validierung
#   - `sklearn.model_selection`: Train-Test Split, Cross-Validation, Grid Search
# - **Preprocessing**: Datenvorverarbeitung
#   - `sklearn.preprocessing`: Feature Scaling, Encoding
# - **Metrics**: Bewertungsmetriken
#   - `sklearn.metrics`: Accuracy, Precision, Recall, F1, MAE, MSE, etc.

# %% [markdown]
#
# ## Die Scikit-Learn API
#
# - Alle Algorithmen folgen einer einheitlichen API:
#   - `fit(X, y)`: Trainiert das Modell mit Trainingsdaten
#   - `predict(X)`: Macht Vorhersagen für neue Daten
#   - `score(X, y)`: Bewertet das Modell (je nach Algorithmus unterschiedlich)
# - Das macht es einfach, verschiedene Algorithmen auszuprobieren
