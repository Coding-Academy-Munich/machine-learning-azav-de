# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Gradientenabstieg</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Die große Frage
#
# - Wir wissen: Das Netz minimiert den Fehler
# - Aber **wie genau** findet es die besten Parameter?
# - Die Antwort: **Gradientenabstieg** (Gradient Descent)

# %% [markdown]
#
# ## Eine Analogie: Abstieg vom Berg
#
# - Stellen Sie sich vor, Sie stehen auf einem Berg
# - Es ist neblig, Sie können nur wenige Meter weit sehen
# - Ihr Ziel: Ins Tal hinabsteigen
# - Wie finden Sie den Weg?

# %% [markdown]
#
# ## Die Strategie
#
# 1. Schauen Sie sich um (im Nebel)
# 2. Finden Sie die steilste Stelle nach unten
# 3. Machen Sie einen Schritt in diese Richtung
# 4. Wiederholen Sie, bis Sie im Tal sind

# %% [markdown]
#
# ## Gradientenabstieg funktioniert genauso!
#
# - **Berg** = Loss-Funktion (Fehler)
# - **Tal** = Minimaler Fehler (bestes Modell)
# - **Steilste Stelle** = Gradient
# - **Schritt** = Parameter-Update

# %% [markdown]
#
# ## Visualisierung: Einfache Fehler-Landschaft
#
# - Stellen wir uns vor, wir haben nur einen Parameter
# - Die x-Achse zeigt den Parameterwert
# - Die y-Achse zeigt den Fehler (Loss)

# %%
from nn_training_essentials_plots import plot_simple_loss_landscape

# %%

# %% [markdown]
#
# ## Schrittweiser Abstieg
#
# - Wir starten irgendwo auf dem "Berg"
# - Bei jedem Schritt gehen wir bergab
# - Der Gradient zeigt uns die Richtung

# %%
from nn_training_essentials_plots import plot_gradient_descent_steps

# %%

# %% [markdown]
#
# ## Was ist der Gradient?
#
# - Der **Gradient** ist die Steigung der Fehler-Funktion
# - Er zeigt in die Richtung des steilsten **Anstiegs**
# - Wir gehen in die **entgegengesetzte** Richtung (bergab)
# - Mathematisch: Eine Ableitung

# %% [markdown]
#
# ## 2D-Beispiel: Zwei Parameter
#
# - Neuronale Netze haben viele Parameter
# - Schauen wir uns zwei Parameter an
# - Die Fehler-Landschaft wird zu einem Gebirge

# %%
from nn_training_essentials_plots import plot_2d_loss_landscape, plot_2d_loss_landscape_3d

# %%

# %%

# %% [markdown]
#
# ## Das Problem: Lokale Minima
#
# - Unsere Analogie hat einen Haken!
# - Was passiert, wenn es mehrere Täler gibt?
# - Sie könnten in einem kleinen Tal landen, nicht im tiefsten

# %% [markdown]
#
# <img src="img/Funtensee-1.jpg" style="width:40%;float:right;margin-left:10px;"/>
#
# ## Beispiel: Der Funtensee
#
# <ul>
#   <li>See in den Berchtesgadener Alpen</li>
#   <li>In einem Tal, das von höheren Bergen<br>umgeben ist</li>
#   <li>Kältester Ort Deutschlands</li>
#   <li>Wenn Sie dort landen, denken Sie vielleicht,<br>Sie sind am tiefsten Punkt</li>
#   <li>Aber Sie sind immer noch 1601 Meter hoch!</li>
# </ul>

# %% [markdown]
#
# <img src="img/Funtensee-2.jpg" style="width:80%;margin-left:10px;"/>

# %%
from nn_training_essentials_plots import plot_funtensee_local_minimum

# %%

# %% [markdown]
#
# ## Lokales vs. Globales Minimum
#
# - **Lokales Minimum**: Ein Tal, aber nicht das tiefste
# - **Globales Minimum**: Das tiefste Tal überhaupt
# - Problem: Gradient Descent kann in lokalem Minimum stecken bleiben
# - Wie der Funtensee: Ein Tal, aber nicht das tiefste

# %% [markdown]
#
# ## Demonstration: Wo man startet, ist wichtig

# %%
from nn_training_essentials_plots import plot_starting_point_matters

# %%

# %% [markdown]
#
# ## Die Lernrate
#
# - Wie groß soll der Schritt sein?
# - Die **Lernrate** (Learning Rate) steuert die Schrittgröße
# - **Zu groß**: Man springt über das Minimum hinweg
# - **Zu klein**: Man braucht sehr lange
# - **Genau richtig**: Effizientes Lernen

# %%
from nn_training_essentials_plots import plot_learning_rate_effects

# %%

# %% [markdown]
#
# ## In der Praxis
#
# - Neuronale Netze haben Hunderte oder Tausende von Parametern
# - Der Gradient zeigt für jeden Parameter die Richtung
# - Moderne Optimierer (wie Adam) passen die Lernrate automatisch an
# - Sie helfen, lokale Minima zu vermeiden

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Gradientenabstieg** = Methode zum Finden des Minimums
# - **Gradient** = Richtung des steilsten Anstiegs
# - Wir gehen in die entgegengesetzte Richtung (bergab)
# - **Lernrate** steuert die Schrittgröße
# - **Lokale Minima** wie der Funtensee sind ein Problem
# - Moderne Optimierer helfen, gute Lösungen zu finden

# %%
