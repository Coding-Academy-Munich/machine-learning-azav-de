# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Decision Trees</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was sind Decision Trees?
#
# - **Baumstruktur** für Entscheidungen
# - Splittet Daten basierend auf Features
# - **Interpretierbar**: Kann als Regeln gelesen werden
# - Basis für Random Forests

# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# %%
# Example data
df = pd.DataFrame({
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'passed': [0, 0, 0, 1, 1, 1, 1, 1]
})

X = df[['hours_studied']]
y = df['passed']

# %%
# TODO: Train decision tree

# %% [markdown]
#
# ## Zusammenfassung
#
# - Decision Trees sind interpretierbar
# - Können nicht-lineare Beziehungen lernen
# - Neigen zu Overfitting (max_depth limitieren!)
# - Basis für Random Forests

# %%
