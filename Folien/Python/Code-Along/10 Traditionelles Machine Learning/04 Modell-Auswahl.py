# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Modell-Auswahl</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Modelle vergleichen
#
# - Verschiedene Algorithmen testen
# - Cross-Validation nutzen
# - Metriken vergleichen
# - Bestes Modell wählen

# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

# %%
