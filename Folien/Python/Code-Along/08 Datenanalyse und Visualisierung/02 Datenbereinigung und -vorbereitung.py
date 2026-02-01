# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Datenbereinigung und -vorbereitung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Daten bereinigen
#
# - Fehlende Werte behandeln
# - Ausreißer erkennen
# - Daten normalisieren
# - Kategorische Variablen kodieren

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# %%
# Example data
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 100],
    'salary': [50000, 60000, 55000, 75000, 1000000],
    'city': ['Berlin', 'Munich', 'Berlin', 'Hamburg', 'Berlin']
})

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - Datenbereinigung ist essentiell
# - pandas macht es einfach
# - scikit-learn für Skalierung und Encoding
# - Pipelines für wiederholbare Prozesse

# %%
