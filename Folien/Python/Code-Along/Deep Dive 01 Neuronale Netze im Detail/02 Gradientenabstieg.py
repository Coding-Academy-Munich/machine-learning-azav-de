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

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from nn_training_plots import (
    plot_simple_loss_landscape, plot_gradient_descent_steps,
    plot_2d_loss_landscape, plot_2d_loss_landscape_3d,
    plot_local_minimum, plot_different_starting_points,
    plot_learning_rates
)

# %%
sns.set_theme(style="darkgrid")

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

# %%
# Create a simple 1D loss function (parabola)
w = np.linspace(-5, 5, 100)
loss = (w - 2) ** 2 + 1


# %%

# %% [markdown]
#
# ## Schrittweises Abstieg

# %%
# Simulate gradient descent on the simple function
def gradient_simple(w):
    """Gradient of (w-2)^2 + 1"""
    return 2 * (w - 2)

# %%
# Gradient descent simulation
w_start = -4
learning_rate = 0.3
steps = 15

w_history = [w_start]
loss_history = [(w_start - 2) ** 2 + 1]

w_current = w_start
for _ in range(steps):
    grad = gradient_simple(w_current)
    w_current = w_current - learning_rate * grad
    w_history.append(w_current)
    loss_history.append((w_current - 2) ** 2 + 1)

# %%

# %% [markdown]
#
# ## Was ist der Gradient?
#
# - Der **Gradient** ist die Steigung der Fehler-Funktion
# - Er zeigt in die Richtung des steilsten **Anstiegs**
# - Wir gehen in die **entgegengesetzte** Richtung (bergab)
# - Mathematisch: Eine Ableitung (oder mehrere bei vielen Parametern)

# %% [markdown]
#
# ## 2D-Beispiel: Zwei Parameter
#
# - Neuronale Netze haben viele Parameter
# - Schauen wir uns zwei Parameter an
# - Die Fehler-Landschaft wird zu einem Gebirge

# %%
# Create a 2D loss landscape
w1_range = np.linspace(-3, 3, 100)
w2_range = np.linspace(-3, 3, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)

# %%
# Simple bowl-shaped loss function
Loss_2d = (W1 - 1) ** 2 + (W2 + 0.5) ** 2 + 0.5

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
# ## Beispiel: Der Funtensee
#
# - Der Funtensee liegt in den Berchtesgadener Alpen
# - Es ist ein Tal, das von höheren Bergen umgeben ist
# - Kältester Ort Deutschlands (kalte Luft sammelt sich)
# - Wenn Sie dort landen, denken Sie vielleicht, Sie sind am tiefsten Punkt
# - Aber es gibt tiefere Täler woanders!

# %%
# Create a loss landscape with local minimum (like Funtensee)
w_landscape = np.linspace(-5, 5, 200)
# Create two valleys using negative Gaussians
# Local valley (Funtensee) at w=-2 (shallower)
# Global valley at w=3 (deeper)
valley_local = -2.0 * np.exp(-((w_landscape + 2) ** 2) / 1.0)
valley_global = -3.0 * np.exp(-((w_landscape - 3) ** 2) / 1.5)
# Add baseline to keep all values positive
loss_with_local = 5.5 + valley_local + valley_global


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
# Gradient for the landscape with local minimum
def gradient_landscape(w):
    """Numerical gradient of the loss landscape"""
    h = 0.01
    w_idx = np.argmin(np.abs(w_landscape - w))
    if w_idx == 0 or w_idx == len(w_landscape) - 1:
        return 0
    return (loss_with_local[w_idx + 1] - loss_with_local[w_idx - 1]) / (2 * h)

# %%
# Two different starting points
def run_gradient_descent(w_start, learning_rate=0.1, steps=100):
    """Run gradient descent from a starting point"""
    w_history = [w_start]
    w_current = w_start

    for _ in range(steps):
        grad = gradient_landscape(w_current)
        w_current = w_current - learning_rate * grad
        w_history.append(w_current)

    return w_history

# %%
# Start from left side (will reach Funtensee)
path_left = run_gradient_descent(-4.0, learning_rate=0.05, steps=100)

# Start from right side (will reach global minimum)
path_right = run_gradient_descent(4.5, learning_rate=0.05, steps=100)

# %%

# %% [markdown]
#
# ## Die Lernrate
#
# - Wie groß soll der Schritt sein?
# - Die **Lernrate** (Learning Rate) steuert die Schrittgröße
# - Zu groß: Man springt über das Minimum hinweg
# - Zu klein: Man braucht sehr lange
# - Genau richtig: Effizientes Lernen

# %%
# Demonstrate different learning rates
path_small = run_gradient_descent(-4.0, learning_rate=0.01, steps=100)
path_good = run_gradient_descent(-4.0, learning_rate=0.05, steps=100)
path_large = run_gradient_descent(-4.0, learning_rate=0.2, steps=100)

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

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - Wie wir das in der Praxis anwenden
# - Epochen, Batches, und mehr
# - Praktische Tipps für das Training

# %%
