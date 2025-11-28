# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>War das wirklich ML?</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ### War das wirklich schon ML?
#
# <table>
# <tr>
#   <td><span style="color: green;">✓</span></td>
#   <td>Erstellen/Verwenden eines Modells</td>
# </tr>
# <tr>
#   <td><span style="color: green;">✓</span></td>
#   <td>Trainieren des Modells mit Daten</td>
# </tr>
# <tr>
#   <td><span style="color: blue;">○</span>
#   </td><td>Analysieren/Visualisieren der Daten</td>
# </tr>
# <tr>
#   <td><span style="color: blue;">○</span></td>
#   <td>Evaluierung des Modells</td>
# </tr>
# <tr>
#   <td><span style="color: green;">✓</span></td>
#   <td>Modell generalisiert gut auf neue Daten</td>
# </tr>
# <tr>
#   <td><span style="color: red;">✗</span></td>
#   <td>Menge der Trainingsdaten</td>
# </tr>
# <tr>
#   <td><span style="color: red;">✗</span></td>
#   <td>Trainingsdaten ohne Fehler/Ausreißer/Rauschen</td>
# </tr>
# <tr>
#   <td><span style="color: red;">✗</span>
#   </td><td>Vorbereiten der Daten für das Training</td>
# </tr>
# <tr>
#   <td><span style="color: red;">✗</span></td>
#   <td>Anzahl der Features</td>
# </tr>
# <tr>
#   <td><span style="color: red;">✗</span></td>
#   <td>Anpassen der Hyperparameter</td>
# </tr>
# </table>

# %% [markdown]
#
# ## Der ML Workflow
#
# <img src="../../../../img/ml-workflow-data.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# Es gibt verschiedene Schritte, die wir bei praktisch jedem ML Problem durchführen müssen:
#
# 1. Daten sammeln, analysieren und vorbereiten
# 2. Ein Modell auswählen und trainieren
# 3. Das Modell/System bewerten und ggf. verbessern
# 4. Das Modell/System in Produktion bringen

# %% [markdown]
#
# ### Überraschende Ergebnisse
#
# - Lineare Regression wird in der Praxis eingesetzt (mit mehreren Features, Feature Engineering, ...)
# - Dieser einfache Mechanismus ist einer der Grundbausteine von Neuronalen Netzen und LLMs
