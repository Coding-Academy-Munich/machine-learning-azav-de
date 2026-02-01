# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Grenzen linearer Modelle</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bisher gelernt haben
#
# - Lineare Regression kann Beziehungen zwischen Features und Labels modellieren
# - Sie funktioniert gut für lineare Zusammenhänge
# - Multivariate Regression kann mehrere Features berücksichtigen

# %% [markdown]
#
# ## Beispiel: Eiscreme-Verkauf
#
# - Erinnerung: Wir haben Eiscreme-Verkauf basierend auf Temperatur und Bewölkung vorhergesagt
# - Das lineare Modell hat gut funktioniert!
# - Die Beziehung war (ungefähr) linear

# %%
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# Laden der Eiscreme-Daten

# %%
try:
    db_path = Path(__file__).parent / "ice_cream_sales.db"
except NameError:
    db_path = Path("ice_cream_sales.db")
connection = sqlite3.connect(str(db_path))

# %%
cursor = connection.cursor()

# %%
cursor.execute("SELECT temperature, cloud_cover, sales FROM sales")

# %%
data = cursor.fetchall()

# %%
connection.close()

# %%
ice_cream_x = [list(row[:2]) for row in data]
ice_cream_y = [row[2] for row in data]

# %% [markdown]
#
# ### Lineare Beziehung mit Temperatur

# %%
from building_nn_azav_plots import plot_ice_cream_temperature

# %%

# %% [markdown]
#
# ## Aber nicht alle Probleme sind linear!
#
# - In der realen Welt sind viele Beziehungen komplexer
# - Lineare Modelle stoßen an ihre Grenzen
# - Schauen wir uns einige Beispiele an

# %% [markdown]
#
# ## Problem 1: Nichtlineare Beziehungen
#
# - Beispiel: Reaktionszeit bei Aufgaben
# - Bei leichten Aufgaben: schnelle Reaktion
# - Bei mittleren Aufgaben: langsamere Reaktion
# - Bei sehr schweren Aufgaben: wieder schneller (weil man aufgibt!)

# %%
# Create synthetic U-shaped data
np.random.seed(42)
x_difficulty = np.linspace(0, 10, 100)
y_reaction = 0.5 * (x_difficulty - 5)**2 + 2 + np.random.normal(0, 0.5, 100)

# %% [markdown]
#
# ### U-förmige Beziehung

# %%
from building_nn_azav_plots import plot_u_shaped_data

# %%

# %% [markdown]
#
# ## Lineare Regression versagt hier
#
# - Eine Gerade kann diese U-Form nicht erfassen
# - Das Modell würde schlechte Vorhersagen machen

# %%
from sklearn.linear_model import LinearRegression

# %% [markdown]
#
# Versuch mit linearer Regression auf U-förmigen Daten

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ### Lineares Modell scheitert

# %%
from building_nn_azav_plots import plot_linear_model_on_u_shaped

# %%

# %% [markdown]
#
# ## Problem 2: Plateaus und Sättigung
#
# - Beispiel: Lernfortschritt beim Üben
# - Am Anfang: schneller Fortschritt
# - Später: der Fortschritt verlangsamt sich
# - Irgendwann: kaum noch Verbesserung (Plateau)

# %%
# Create sigmoid-shaped learning curve data
x_practice = np.linspace(0, 100, 100)
y_skill = 100 / (1 + np.exp(-(x_practice - 50) / 10)) + np.random.normal(0, 2, 100)

# %% [markdown]
#
# ### Lernkurve mit Plateau

# %%
from building_nn_azav_plots import plot_learning_plateau

# %%

# %% [markdown]
#
# ## Problem 3: XOR-Problem
#
# - Ein klassisches Problem für lineare Modelle
# - Zwei Features: x₁ und x₂ (jeweils 0 oder 1)
# - Label: 1 wenn x₁ ≠ x₂, sonst 0
# - Eine Gerade kann diese Daten nicht trennen!

# %%
# Create XOR data
xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_y = np.array([0, 1, 1, 0])

# %% [markdown]
#
# ### Visualisierung des XOR-Problems

# %%
from building_nn_azav_plots import plot_xor_problem

# %%

# %% [markdown]
#
# ## Was brauchen wir?
#
# - Ein flexibleres Modell
# - Eines, das gebogene Linien lernen kann
# - Eines, das komplexe Muster erkennen kann
# - **Neuronale Netze!**

# %% [markdown]
#
# ## Die Grundidee
#
# - Was wäre, wenn wir mehrere lineare Modelle kombinieren könnten?
# - Jedes lernt einen Teil des Problems
# - Zusammen können sie komplexere Muster lernen

# %% [markdown]
#
# ## Beispiel: Zwei Geraden statt einer
#
# - Eine Gerade für niedrige Werte
# - Eine Gerade für hohe Werte
# - Kombiniert können sie eine Kurve approximieren

# %%
x_demo = np.linspace(0, 10, 100)

# %% [markdown]
#
# Zwei lineare Stücke

# %%
y_piece1 = 0.5 * x_demo - 1.5

# %%
y_piece2 = -0.5 * x_demo + 3.5

# %%
y_combined = y_piece1 + y_piece2

# %%
from building_nn_azav_plots import plot_combining_linear_pieces

# %%

# %% [markdown]
#
# - **Das funktioniert nicht!**
# - Die Kombination von linearen Modellen ist immer noch linear
# - Wir brauchen ein Möglichkeit, ein Modell für einen bestimmten Bereich zu
#   aktivieren

# %% [markdown]
#
# - Was passiert, wenn wir nur den positiven Teil unserer Modelle nehmen?

# %%
x_demo = np.linspace(0, 10, 100)

# %%
y_piece1 = np.maximum(0, 0.5 * x_demo - 1.5)

# %%
y_piece2 = np.maximum(0, -0.5 * x_demo + 3.5)

# %%
y_combined = y_piece1 + y_piece2

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - Lineare Modelle sind begrenzt auf gerade Linien
# - Viele reale Probleme sind nichtlinear
# - Wir brauchen flexiblere Modelle
# - Neuronale Netze kombinieren einfache Komponenten zu komplexen Lösungen

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - Wir bauen unser erstes neuronales Netz
# - Wir verstehen, wie Neuronen funktionieren
# - Wir sehen, wie man sie kombiniert

# %%
