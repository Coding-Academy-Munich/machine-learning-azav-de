# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Warum Train/Test-Split?</b>
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
# - Das folgende Modell merkt sich einfach die Trainingsdaten und gibt sie bei
#   der Vorhersage wieder aus.
# - Es generalisiert nicht.

# %%

# %% [markdown]
#
# ### Ohne Train/Test Split

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Ist das Modell wirklich so gut, wie unsere Evaluierung sagt?

# %% [markdown]
#
# ### Mit Train/Test Split

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# - Das Modell ist sehr schlecht!
# - Es hat sich nur die Trainingsdaten gemerkt und kann auf den
#   Testdaten nichts vorhersagen.
# - Das nennt man Overfitting.
