# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Von Neuronalen Netzen zu Large Language Models</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bisher gelernt haben
#
# - Neuronale Netze können komplexe Muster lernen
# - Sie bestehen aus Neuronen, Gewichten und Aktivierungsfunktionen
# - Durch Training lernen sie, Vorhersagen zu machen
# - Wir haben sie für Zahlen und einfache Klassifikation genutzt
#
# **Aber**: Neuronale Netze können noch viel mehr!

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Neuronale Netze auf Text
#
# - **Frage**: Können neuronale Netze auch mit Text arbeiten?
# - **Problem**: Netze verarbeiten nur Zahlen, nicht Wörter
# - **Lösung**: Wörter in Zahlen umwandeln!
# - Das nennt man **Embeddings** (Einbettungen)

# %% [markdown]
#
# ## Was sind Embeddings?
#
# - Jedes Wort wird als Vektor (Liste von Zahlen) dargestellt
# - Ähnliche Wörter haben ähnliche Vektoren
# - **Beispiel**:
#   - "König" und "Königin" sind nahe beieinander
#   - "Hund" und "Katze" sind nahe beieinander
#   - "König" und "Tisch" sind weit auseinander

# %% [markdown]
#
# Vereinfachte Wort-Embeddings (2D zur Visualisierung)
# In der Realität haben Embeddings typischerweise 300-1000 Dimensionen

# %%
word_embeddings = {
    'king': np.array([0.8, 0.9]),
    'queen': np.array([0.75, 0.85]),
    'man': np.array([0.7, 0.3]),
    'woman': np.array([0.65, 0.25]),
    'dog': np.array([-0.4, 0.7]),
    'cat': np.array([-0.35, 0.65]),
    'table': np.array([-0.8, -0.6]),
    'chair': np.array([-0.75, -0.65])
}

# %%

# %% [markdown]
#
# ## Attention: Die Geheimwaffe
#
# - **Problem**: Nicht alle Wörter im Satz sind gleich wichtig
# - **Lösung**: **Attention-Mechanismus**
# - Das Netzwerk lernt, auf wichtige Teile zu "achten"
# - Wie wenn Sie einen Text lesen und wichtige Wörter hervorheben

# %% [markdown]
#
# ## Beispiel: Attention
#
# **Satz**: "Der Hund jagte die Katze durch den Park"
#
# - Wenn wir über "jagte" nachdenken:
#   - Wichtig: "Hund" (wer jagt?)
#   - Wichtig: "Katze" (wen wird gejagt?)
#   - Weniger wichtig: "der", "die", "durch"
#
# Attention gewichtet automatisch, welche Wörter wichtig sind!

# %%
# Simplified attention visualization
sentence = "The dog chased the cat".split()
# Attention weights (simplified - showing which words matter for "chased")
attention_weights = np.array([0.1, 0.4, 1.0, 0.1, 0.35])

# %%

# %% [markdown]
#
# ## Von klein zu groß: Die Größenrevolution
#
# - Unsere neuronalen Netze: ~1.000-10.000 Parameter
# - Moderne neuronale Netze: Millionen von Parametern
# - **Large Language Models**: **Milliarden** von Parametern!
#
# | Modell | Parameter | Jahr |
# |--------|-----------|------|
# | GPT-2 | 1.5 Milliarden | 2019 |
# | GPT-3 | 175 Milliarden | 2020 |
# | GPT-4 | ~1 Billion (geschätzt) | 2023 |

# %%
# Visualization of parameter scales
models = ['Unsere Netze\nOur Networks', 'GPT-2', 'GPT-3', 'GPT-4\n(geschätzt)']
params = [10000, 1.5e9, 175e9, 1e12]  # in actual numbers
params_display = [10, 1500, 175000, 1000000]  # in millions for display

# %%

# %% [markdown]
#
# ## Warum so groß?
#
# - **Mehr Parameter** = mehr Kapazität zum Lernen
# - LLMs werden auf **riesigen Textmengen** trainiert
#   - Bücher, Wikipedia, Websites, Code, ...
#   - Milliarden von Wörtern!
# - Sie lernen:
#   - Grammatik und Syntax
#   - Fakten und Wissen
#   - Schreibstile und Muster
#   - Logik und Schlussfolgerungen

# %% [markdown]
#
# ## Was können LLMs?
#
# - **Text generieren**: Geschichten, Artikel, Code schreiben
# - **Fragen beantworten**: Wissen abrufen und erklären
# - **Übersetzen**: Zwischen Sprachen übersetzen
# - **Zusammenfassen**: Lange Texte kurz zusammenfassen
# - **Code schreiben**: Programme in verschiedenen Sprachen
# - **Konversation**: Natürliche Unterhaltungen führen

# %% [markdown]
#
# ## Beispiele für LLMs
#
# - **GPT-4** (OpenAI): Sehr leistungsfähig, vielseitig
# - **Claude** (Anthropic): Gut bei langen Texten, hilfreich
# - **Llama 2** (Meta): Open Source, läuft lokal
# - **Gemini** (Google): Multimodal (Text + Bilder)
#
# Alle basieren auf denselben Grundprinzipien:
# Neuronale Netze + Embeddings + Attention!

# %% [markdown]
#
# ## Die Kernbotschaft
#
# **LLMs sind neuronale Netze wie die, die Sie gebaut haben!**
#
# - Dieselben Grundprinzipien: Gewichte, Training, Muster lernen
# - Nur **viel größer** und auf **viel mehr Text** trainiert
# - Kombiniert mit **cleveren Techniken** (Embeddings, Attention)
# - Das Ergebnis: Systeme, die Sprache verstehen und generieren können
#
# **Im nächsten Abschnitt**: Lernen wir, wie man LLMs nutzt!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Embeddings**: Wörter als Zahlen → Netze können damit arbeiten
# - **Attention**: Konzentrieren auf wichtige Teile des Textes
# - **Größe**: Von Tausenden zu Milliarden von Parametern
# - **Training**: Auf riesigen Textmengen lernen
# - **Fähigkeiten**: Text verstehen, generieren, übersetzen, ...
#
# **Sie haben schon die Grundlagen!** Jetzt nutzen wir große Modelle.

# %%
