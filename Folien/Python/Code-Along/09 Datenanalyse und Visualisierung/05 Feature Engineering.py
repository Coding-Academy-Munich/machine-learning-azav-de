# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Feature Engineering</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was sind Features?
#
# - **Features** = Eingabevariablen für ML-Modelle
# - Aus vorhandenen Daten **neue Features** erstellen
# - **Domain Knowledge** nutzen
# - Modell-Performance verbessern

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# %% [markdown]
#
# ## Feature-Typen
#
# - Polynomial features: x² , x³
# - Interaction terms: x₁ * x₂
# - Time-based: Jahr, Monat, Wochentag
# - Text features: Länge, Wortanzahl
# - Domain-specific: Branchenspezifisch

# %%
