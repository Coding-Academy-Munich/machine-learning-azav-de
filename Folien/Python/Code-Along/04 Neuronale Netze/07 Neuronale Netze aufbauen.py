# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Neuronale Netze aufbauen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bisher haben
#
# - Ein einzelnes Neuron
# - Nimmt Eingaben, wendet Gewichte an
# - Aktivierungsfunktion macht es nichtlinear
# - Aber: ein Neuron ist immer noch begrenzt

# %% [markdown]
#
# ## Die Idee: Teamwork!
#
# - Ein Neuron = ein Entscheidungsträger
# - Viele Neuronen = ein Team von Entscheidungsträgern
# - Jedes Neuron lernt einen anderen Aspekt
# - Zusammen können sie komplexe Probleme lösen

# %% [markdown]
#
# ## Analogie: Ein Unternehmen
#
# - **Mitarbeiter** (untere Ebene): Sehen die Rohdaten
# - **Manager** (mittlere Ebene): Kombinieren Informationen von Mitarbeitern
# - **Direktor** (obere Ebene): Trifft die finale Entscheidung
#
# Jede Ebene baut auf der vorherigen auf!

# %% [markdown]
#
# ## Multi-Layer Perceptron (MLP)
#
# - **Eingabeschicht**: Die Features (unsere Daten)
# - **Versteckte Schicht(en)**: Neuronen, die Muster lernen
# - **Ausgabeschicht**: Die Vorhersage
#
# "Versteckt" = wir sehen nicht direkt, was sie machen

# %% [markdown]
#
# ## Einfaches MLP Diagramm
#
# <img src="../../../../img/mlp-structure.png" alt="MLP Struktur" style="width:50%;float:right;margin-left:5px;">
#
# - 2 Neuronen in der versteckten Schicht
# - Jedes erhält alle 3 Eingaben
# - Die Ausgabe kombiniert alle 2

# %% [markdown]
#
# ## Wie viele Neuronen in der versteckten Schicht?
#
# - **Zu wenige**: Netz kann nicht genug lernen
# - **Zu viele**: Netz lernt die Trainingsdaten auswendig (Overfitting)
# - **Richtig**: Experimentieren und testen!

# %% [markdown]
#
# ## Typische Größen
#
# - Kleine Probleme: 5-500 Neuronen
# - Mittlere Probleme: 500-20,000 Neuronen
# - Große Probleme: Tausende bis Millionen
#
# Aber: Mehr ist nicht immer besser!

# %% [markdown]
#
# ## Demonstration: XOR Problem lösen
#
# - Erinnern Sie sich: Lineare Regression konnte das nicht
# - Schauen wir, ob ein MLP es kann!

# %%
import numpy as np

# %% [markdown]
#
# Erzeuge XOR Daten

# %%
xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_y = np.array([0, 1, 1, 0])

# %% [markdown]
#
# ### Visualisierung des XOR-Problems

# %%
from building_nn_azav_plots import plot_xor_problem_mlp


# %%

# %% [markdown]
#
# ## Manuelles MLP für XOR
#
# <img src="../../../../img/xor-mlp.png" alt="XOR MLP" style="width:50%;float:right;margin-left:5px;">
#
# - Wir können ein MLP mit 2 Neuronen<br>in der versteckten Schicht bauen
# - Diese können das XOR-Problem lösen!
# - Schauen wir uns an, wie

# %%
def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

# %%

# %% [markdown]
#
# ## Testen des manuellen MLPs

# %%
print("x₁  x₂  |  h₁   h₂      |  Output (should be XOR)")
print("-" * 50)
for x1, x2 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    h1, h2, output = manual_xor_mlp(x1, x2)
    print(f"{x1}   {x2}   |  {h1:.2f}  {h2:.2f}  |  {output:.2f}")

# %% [markdown]
#
# ## Mehrere versteckte Schichten
#
# - Wir können auch mehrere versteckte Schichten haben
# - **Tiefe Netze** = viele Schichten
# - Jede Schicht lernt komplexere Muster

# %% [markdown]
#
# ## Tiefes Netzwerk Diagramm
#
# <img src="../../../../img/deep-net-structure.png" alt="Deep Network Struktur" style="width:50%;float:right;margin-left:5px;">
#
# ```
# Input  →  Hidden 1  →  Hidden 2  →  Output
# ```
#
# - Erste Schicht: einfache Muster
# - Zweite Schicht: komplexere Muster
# - Ausgabe: finale Entscheidung

# %% [markdown]
#
# ## Visualisierung: Was lernt jede Schicht?
#
# - Bei Bildern (kommen später):
#   - Schicht 1: Kanten und Linien
#   - Schicht 2: Formen und Texturen
#   - Schicht 3: Teile von Objekten
#   - Schicht 4: Ganze Objekte

# %% [markdown]
#
# ## Demonstration: Entscheidungsgrenzen
#
# - Schauen wir uns an, wie ein MLP Daten trennt
# - Vergleichen mit linearer Regression

# %% [markdown]
#
# Erzeugen nicht-linear trennbarer Daten (kreisförmig)

# %%
np.random.seed(42)

# %%
n_samples = 200

# %% [markdown]
#
# Innerer Kreis (Klasse 0)

# %%
r_inner = np.random.uniform(0, 1, n_samples // 2)

# %%
theta_inner = np.random.uniform(0, 2 * np.pi, n_samples // 2)

# %%
x_inner = r_inner * np.cos(theta_inner)

# %%
y_inner = r_inner * np.sin(theta_inner)

# %% [markdown]
#
# Äußerer Kreis (Klasse 1)

# %%
r_outer = np.random.uniform(1.5, 2.5, n_samples // 2)

# %%
theta_outer = np.random.uniform(0, 2 * np.pi, n_samples // 2)

# %%
x_outer = r_outer * np.cos(theta_outer)

# %%
y_outer = r_outer * np.sin(theta_outer)

# %% [markdown]
#
# Kombinieren der Daten

# %%
X_circles = np.vstack([
    np.column_stack([x_inner, y_inner]),
    np.column_stack([x_outer, y_outer])
])

# %%
y_circles = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# %% [markdown]
#
# ### Visualisierung der kreisförmigen Daten

# %%
from building_nn_azav_plots import plot_circular_data

# %%

# %% [markdown]
#
# ## Lineare Regression scheitert hier

# %%
from sklearn.linear_model import LogisticRegression

# %% [markdown]
#
# Versuch mit logistischer Regression (lineare Entscheidungsgrenze)

# %%

# %% [markdown]
#
# Gitter für Entscheidungsgrenze erstellen

# %%
h = 0.1

# %%
x_min, x_max = X_circles[:, 0].min() - 0.5, X_circles[:, 0].max() + 0.5

# %%
y_min, y_max = X_circles[:, 1].min() - 0.5, X_circles[:, 1].max() + 0.5

# %%
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# %%

# %%

# %% [markdown]
#
# ### Lineares Modell: Schlechte Trennung

# %%
from building_nn_azav_plots import plot_linear_decision_boundary

# %%

# %% [markdown]
#
# ## MLP kann es!

# %%
from sklearn.neural_network import MLPClassifier

# %% [markdown]
#
# Neuronales Netz mit versteckten Schichten

# %%

# %%

# %%

# %% [markdown]
#
# ### Neuronales Netz: Exzellente Trennung!

# %%
from building_nn_azav_plots import plot_mlp_decision_boundary

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - MLPs kombinieren viele Neuronen in Schichten
# - Jede Schicht lernt komplexere Muster
# - Versteckte Schichten = "Team von Entscheidungsträgern"
# - Können nichtlineare Probleme lösen

# %% [markdown]
#
# ## Was noch fehlt
#
# - Wir haben Netzwerke "von Hand" gebaut
# - Aber wie lernen sie die Gewichte?
# - Das sehen wir in der nächsten Lektion!

# %%
