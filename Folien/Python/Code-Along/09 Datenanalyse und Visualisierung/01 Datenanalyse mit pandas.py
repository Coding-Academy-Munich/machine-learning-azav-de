# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Datenanalyse mit pandas</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was ist pandas?
#
# - **Bibliothek** für Datenanalyse in Python
# - **DataFrame**: Tabelle mit Zeilen und Spalten
# - **Standard-Werkzeug** für Data Science
# - Ähnlich wie Excel, aber viel mächtiger!

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## DataFrame erstellen

# %%
# TODO: Create DataFrame from dictionary

# %%

# %% [markdown]
#
# ## Daten laden
#
# pandas kann viele Formate lesen:
# - CSV: `pd.read_csv()`
# - Excel: `pd.read_excel()`
# - JSON: `pd.read_json()`
# - SQL: `pd.read_sql()`

# %% [markdown]
#
# ## Erste Dateninspektion
#
# - `df.head()`: Erste Zeilen anzeigen
# - `df.info()`: Datentypen und fehlende Werte
# - `df.describe()`: Statistische Zusammenfassung
# - `df.shape`: Anzahl Zeilen und Spalten

# %%

# %%

# %%

# %% [markdown]
#
# ## Fehlende Werte behandeln

# %%
# Create DataFrame with missing values
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, 12]
})

# %%

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - pandas: Standard für Datenanalyse
# - DataFrame: Tabellen-Struktur
# - Daten laden, inspizieren, bereinigen
# - Viele Funktionen für EDA
#
# **Nächster Schritt**: Datenbereinigung und -vorbereitung!

# %%
