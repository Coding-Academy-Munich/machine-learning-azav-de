# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Was sind Large Language Models?</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Die Revolution der letzten Jahre
#
# - **ChatGPT** (Ende 2022): Über 100 Millionen Nutzer in 2 Monaten
# - **Large Language Models (LLMs)** überall
# - Können Texte verstehen, generieren, übersetzen, zusammenfassen
# - Aber was genau sind sie?

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from what_are_llms_azav_plots import (plot_llm_timeline, plot_llm_capabilities,
                        plot_training_data_sources, plot_training_stages,
                        plot_emergent_abilities)

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Definition: Large Language Model
#
# - **Large**: Milliarden von Parametern
# - **Language**: Auf Text trainiert
# - **Model**: Neuronales Netz (Transformer)
# - Trainiert auf riesigen Textmengen (Bücher, Websites, etc.)
# - Lernt Muster, Grammatik, Fakten, Logik

# %% [markdown]
#
# ## Bekannte LLMs

# %%

# %% [markdown]
#
# ## Was können LLMs?
#
# - **Text-Generierung**: Artikel, Geschichten, Code schreiben
# - **Übersetzung**: Zwischen Sprachen übersetzen
# - **Zusammenfassung**: Lange Texte zusammenfassen
# - **Fragen beantworten**: Auf Wissensfragen antworten
# - **Code schreiben**: Programmier-Aufgaben lösen
# - **Konversation**: Natürliche Gespräche führen

# %%

# %% [markdown]
#
# ## Wie groß ist "Large"?
#
# - **GPT-2** (2019): 1.5 Milliarden Parameter
# - **GPT-3** (2020): 175 Milliarden Parameter
# - **GPT-4** (2023): ~1 Billion Parameter (geschätzt)
# - Zum Vergleich:
#   - Menschliches Gehirn: ~86 Milliarden Neuronen
#   - Aber: Parameter ≠ Neuronen!

# %% [markdown]
#
# ## Training eines LLM
#
# - **Riesige Datenmengen**: Terabytes von Text
# - **Rechenleistung**: Tausende GPUs über Monate
# - **Kosten**: Millionen von Dollar
# - **Ziel**: Nächstes Wort vorhersagen
# - "The cat sat on the ___" → "mat"

# %%

# %% [markdown]
#
# ## Pre-Training vs. Fine-Tuning
#
# - **Pre-Training**: Auf allgemeinen Daten trainieren
#   - Lernt Sprache, Fakten, Logik
#   - Sehr teuer
# - **Fine-Tuning**: Auf spezielle Aufgaben anpassen
#   - Konversation, Instruktionen befolgen
#   - Viel günstiger

# %%

# %% [markdown]
#
# ## Emergent Abilities (?)
#
# Angeblich:
#
# - Bei einer gewissen Größe: **Neue Fähigkeiten tauchen auf**
# - Kleinere Modelle können es nicht
# - Größere Modelle plötzlich schon!
# - Beispiele:
#   - Mathematik
#   - Logisches Schließen
#   - Multi-Step-Reasoning
#
# Allerdings umstritten!

# %%

# %% [markdown]
#
# ## Limitierungen von LLMs
#
# - **Halluzinationen**: Erfinden manchmal Fakten
# - **Wissens-Cutoff**: Wissen nur bis zu einem bestimmten Datum
# - **Kein echtes Verständnis**: Muster-Matching, nicht "Denken"
# - **Bias**: Vorurteile aus Trainingsdaten
# - **Rechenintensiv**: Brauchen viel Energie

# %% [markdown]
#
# ## Zusammenfassung
#
# - **LLMs**: Riesige neuronale Netze für Text
# - **Basierend auf Transformers** (Attention-Mechanismus)
# - **Trainiert auf Milliarden von Wörtern**
# - **Können viele Aufgaben**: Schreiben, Übersetzen, Fragen beantworten
# - **Emergent Abilities**: Neue Fähigkeiten bei größeren Modellen
# - **Limitierungen**: Halluzinationen, Bias, kein echtes Verständnis

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - **Wie funktionieren LLMs** genauer?
# - Text-Generierung Schritt für Schritt
# - Der Transformer-Mechanismus (vereinfacht)

# %%
