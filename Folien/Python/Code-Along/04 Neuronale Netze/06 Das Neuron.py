# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Das Neuron</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Unser lineares Modell - noch einmal betrachtet
#
# - Lineare Regression: $y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$
# - Nimmt Features ($x_1, x_2, ...$)
# - Multipliziert sie mit Gewichten ($w_1, w_2, ...$)
# - Addiert einen Bias ($b$)
# - Gibt eine Vorhersage aus

# %% [markdown]
#
# ## Das ist (fast schon) ein Neuron!
#
# - Ein künstliches Neuron funktioniert genauso
# - Es ist wie eine "Entscheidungszelle"
# - Nimmt Eingaben, verarbeitet sie, gibt eine Ausgabe
# - Aber...

# %% [markdown]
#
# ## Struktur eines Neurons
#
# <img src="../../../../img/neuron-structure.png" alt="Neuron Structure" style="width:50%;float:right;">
#
# - **Eingaben**: Die Features (x₁, x₂, ...)
# - **Gewichte**: Wie wichtig ist jede Eingabe?
# - **Bias**: Ein Schwellenwert
# - **Ausgabe**: Das Ergebnis

# %% [markdown]
#
# ## Der wichtige Unterschied!
#
# - Bei linearer Regression: direkte Berechnung
# - Bei Neuronen: eine zusätzliche **Aktivierungsfunktion**
# - Diese Funktion macht Neuronen viel mächtiger

# %% [markdown]
#
# ## Was ist eine Aktivierungsfunktion?
#
# - Nach der gewichteten Summe
# - Wird das Ergebnis durch eine Funktion "gefiltert"
# - Diese Funktion entscheidet: "Wie stark soll die Ausgabe sein?"

# %% [markdown]
#
# ## Aktivierungsfunktion 1: Sigmoid
#
# - Wandelt jede Zahl in einen Wert zwischen 0 und 1 um
# - Wie ein **Dimmer-Schalter** für Licht
# - Sehr negative Zahlen → fast 0
# - Sehr positive Zahlen → fast 1
# - Zahlen um 0 → etwa 0.5

# %% [markdown]
#
# Kurze Unterbrechung zum Installieren von Paketen...

# %%
import install_packages  # type: ignore

# %%
from building_nn_azav_plots import (
    plot_neuron_response,
    plot_relu,
    plot_sigmoid_vs_relu,
    plot_sigmoid,
    relu,
    sigmoid,
)


# %% [markdown]
#
# ### Visualisierung der Sigmoid-Funktion

# %%

# %% [markdown]
#
# ## Warum ist das nützlich?
#
# - Begrenzt die Ausgabe
# - Glättet extreme Werte
# - Macht das Modell "sanfter"

# %% [markdown]
#
# ## Aktivierungsfunktion 2: ReLU
#
# - ReLU = **Re**ctified **L**inear **U**nit
# - Wie ein **Einwegventil**
# - Negative Zahlen → 0
# - Positive Zahlen → bleiben unverändert
# - Am häufigsten verwendet!

# %% [markdown]
#
# ### Visualisierung der ReLU-Funktion

# %%

# %% [markdown]
#
# ## ReLU: Einfach aber effektiv
#
# - Sehr schnell zu berechnen
# - Hilft beim Training großer Netze
# - Verhindert bestimmte Probleme beim Lernen

# %% [markdown]
#
# ## Vergleich: Sigmoid vs ReLU
#
# - **Sigmoid**: Glatt, begrenzt, wie ein Dimmer
# - **ReLU**: Einfach, unbegrenzt für positive Werte, wie ein Ventil

# %% [markdown]
#
# ### Vergleich der Aktivierungsfunktionen

# %%

# %% [markdown]
#
# ## Ein Neuron in Aktion
#
# - Schauen wir uns ein konkretes Beispiel an
# - Ein Neuron mit 2 Eingaben
# - ReLU Aktivierungsfunktion

# %%
def simple_neuron(x1, x2, w1=0.5, w2=-0.3, bias=1.0):
    """A simple neuron with 2 inputs and ReLU activation"""
    # Weighted sum
    z = w1 * x1 + w2 * x2 + bias
    # Apply ReLU activation
    output = relu(z)
    return z, output

# %% [markdown]
#
# ## Visualisierung der Neuron-Ausgabe
#
# - Wie reagiert das Neuron auf verschiedene Eingaben?

# %%

# %% [markdown]
#
# ## Warum Aktivierungsfunktionen wichtig sind
#
# - **Ohne** Aktivierungsfunktion: nur lineare Kombinationen möglich
# - **Mit** Aktivierungsfunktion: nichtlineare Muster möglich
# - Das ist der Schlüssel zur Flexibilität neuronaler Netze!

# %% [markdown]
#
# ## Demonstration: Mit vs. Ohne Aktivierung
#
# - Mehrere Neuronen ohne Aktivierung = immer noch linear
# - Mehrere Neuronen mit Aktivierung = nichtlinear!

# %% [markdown]
#
# ### Lineare Funktionen kombiniert bleiben linear

# %%
from building_nn_azav_plots import plot_combining_linear

# %%

# %% [markdown]
#
# ### Mit Aktivierung wird es nichtlinear!

# %%
from building_nn_azav_plots import plot_combining_relu

# %%

# %% [markdown]
#
# ## Zusammenfassung: Das Neuron
#
# - Ein Neuron = gewichtete Summe + Aktivierungsfunktion
# - Aktivierungsfunktionen ermöglichen Nichtlinearität
# - **Sigmoid**: wie ein Dimmer (0 bis 1)
# - **ReLU**: wie ein Ventil (0 oder positiv)

# %% [markdown]
#
# ## Nächster Schritt
#
# - Ein einzelnes Neuron ist noch begrenzt
# - Aber was ist, wenn wir viele Neuronen kombinieren?
# - Das machen wir in der nächsten Lektion!

# %%
