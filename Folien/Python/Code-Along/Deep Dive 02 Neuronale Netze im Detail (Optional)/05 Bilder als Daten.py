# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Bilder als Daten</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was sind Bilder für Computer?
#
# - Für uns: Visuelle Information
# - Für Computer: **Zahlen-Arrays**
# - Jedes Pixel ist eine Zahl
# - Schauen wir uns das genauer an!

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Ein einfaches Beispiel: 8x8 Pixel
#
# - Laden wir einen kleinen Datensatz mit handgeschriebenen Ziffern
# - Jede Ziffer ist nur 8×8 Pixel groß
# - Perfekt, um zu verstehen, wie Bilder gespeichert werden

# %%
# Load digits dataset
digits = load_digits()

# %%
# Look at the first image
first_image = digits.images[0]
first_label = digits.target[0]

# %% [markdown]
#
# ## Das Bild als Array

# %%

# %%

# %% [markdown]
#
# ## Visualisierung

# %%
from cnn_deep_dive_plots import plot_digit, plot_multiple_digits, plot_rgb_image, plot_original_vs_normalized, plot_label_distribution

# %%

# %% [markdown]
#
# ## Graustufenbilder
#
# - Jedes Pixel hat einen Wert zwischen 0 und 255
# - 0 = Schwarz
# - 255 = Weiß
# - Dazwischen = Graustufen
# - Also: 256 mögliche Werte pro Pixel

# %% [markdown]
#
# ## Mehrere Beispiele

# %%

# %% [markdown]
#
# ## Größere Bilder
#
# - 8×8 Pixel sind sehr klein
# - Realistische Bilder sind größer
# - Zum Beispiel: 28×28 (MNIST), 224×224 (Standard für viele Modelle)
# - Mehr Pixel = Mehr Details = Mehr Daten

# %% [markdown]
#
# ## Farbbilder: RGB
#
# - Bisher: Graustufenbilder (1 Wert pro Pixel)
# - Farbbilder: **3 Werte pro Pixel**
# - **R**ed, **G**reen, **B**lue
# - Jede Farbe: 0-255

# %% [markdown]
#
# ## RGB-Beispiel erstellen

# %%
# Create a small RGB image manually
rgb_image = np.zeros((5, 5, 3), dtype=np.uint8)

# %%
# Make top-left corner red
rgb_image[0:2, 0:2, 0] = 255  # R channel

# Make top-right corner green
rgb_image[0:2, 3:5, 1] = 255  # G channel

# Make bottom-left corner blue
rgb_image[3:5, 0:2, 2] = 255  # B channel

# Make center yellow (Red + Green)
rgb_image[2, 2, 0] = 255  # R
rgb_image[2, 2, 1] = 255  # G

# %% [markdown]
#
# ## RGB-Bild visualisieren

# %%

# %% [markdown]
#
# ## Form der Arrays
#
# - **Graustufenbild**: (Höhe, Breite)
#   - Beispiel: (28, 28) für 28×28 Pixel
# - **RGB-Bild**: (Höhe, Breite, 3)
#   - Beispiel: (28, 28, 3) für 28×28 Pixel mit RGB
# - Die 3 = Anzahl der Farbkanäle

# %% [markdown]
#
# ## Bilder als Input für Machine Learning
#
# - Machine Learning braucht Zahlen → Perfekt!
# - Aber: Bilder haben viele Pixel
# - 28×28 = 784 Pixel (Features)
# - 224×224×3 = 150.528 Features!
# - Das ist eine Herausforderung

# %% [markdown]
#
# ## Daten für ML vorbereiten

# %%
# Flatten image to 1D array
flat_image = first_image.flatten()

# %%

# %% [markdown]
#
# ## Normalisierung
#
# - Pixel-Werte: 0-255
# - Oft besser für ML: 0-1
# - Einfach durch Division mit 255
# - Warum? Neuronale Netze lernen besser mit kleineren Zahlen

# %%
# Normalize to 0-1 range
normalized_image = first_image / 15.0

# %%

# %% [markdown]
#
# ## Vergleich: Vorher und Nachher

# %%

# %% [markdown]
#
# ## Der gesamte Datensatz
#
# - Wir haben viele Bilder
# - Alle müssen vorbereitet werden
# - Form: (Anzahl Bilder, Höhe, Breite)

# %%

# %% [markdown]
#
# ## Verteilung der Labels

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - Bilder sind **Arrays von Zahlen**
# - Graustufenbilder: (Höhe, Breite)
# - RGB-Bilder: (Höhe, Breite, 3)
# - Pixel-Werte: Typischerweise 0-255
# - Normalisierung: Oft auf 0-1 skaliert
# - Für ML: Bilder werden oft "flachgeklopft" (flattened)
# - Viele Features → Herausforderung für Standard-ML

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - Spezielle Netze für Bilder: **Convolutional Neural Networks**
# - Wie sie mit vielen Pixeln umgehen
# - Wie sie Muster in Bildern erkennen

# %%
