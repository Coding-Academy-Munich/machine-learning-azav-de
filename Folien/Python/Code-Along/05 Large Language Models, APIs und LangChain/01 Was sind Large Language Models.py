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
# ## Wie funktionieren LLMs?
#
# **Kernprinzip: Nächstes Token vorhersagen**
#
# - LLM sieht: "Der Hund jagt die ___"
# - LLM berechnet Wahrscheinlichkeiten für das nächste Wort
# - "Katze" (30%), "Maus" (25%), "Ball" (15%), ...
# - LLM wählt (meistens) das wahrscheinlichste Wort
#
# **Wichtig**: LLMs "denken" nicht - sie sagen Muster vorher!

# %% [markdown]
#
# ## Token für Token
#
# So entsteht Text:
#
# 1. "Der" → "Der Hund" (wahrscheinlichstes nächstes Wort)
# 2. "Der Hund" → "Der Hund jagt" (wahrscheinlichstes nächstes Wort)
# 3. "Der Hund jagt" → "Der Hund jagt die" → ...
#
# Das LLM generiert **ein Wort nach dem anderen**, basierend auf allem,
# was vorher kam.
#
# Das erklärt auch, warum LLMs manchmal "Unsinn" erzählen: Sie folgen
# statistisch wahrscheinlichen Mustern, nicht Fakten!

# %% [markdown]
#
# ## Bekannte LLMs
#
# - OpenAI: GPT (ChatGPT/Codex; 3, 4, 5, 5.1)
# - Anthropic: Claude (Haiku/Sonnet/Opus; 3, 4, 4.5)
# - Google: Gemini (2.5, 3)
# - Meta: Llama (2, 3, 4)
# - Mistral: (Ministral, Mixtral, Small/Medium/Large; 2, 3)
# - DeepSeek: (V3, R1, V3.2, V3.2 Speciale)
# - MoonshotAI: Kimi K2
# - Z.AI: GLM (4, 4.5, 4.6)
# - Qwen: (2.5, 3)

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

# %% [markdown]
#
# ## Kontext-Fenster (Context Window)
#
# **Wie viel Text kann ein LLM auf einmal "sehen"?**
#
# - **Kontext-Fenster**: Maximale Textmenge für eine Anfrage
# - Gemessen in **Tokens** (~ Wortteile)
# - Faustregel: **1 Token ≈ 0,75 Wörter** (oder 3-4 Zeichen)
#
# | Jahr | Typisches Kontext-Fenster |
# |------|---------------------------|
# | 2023 | 4.000 - 8.000 Tokens |
# | 2024 | 128.000 - 200.000 Tokens |
# | 2025 | Bis zu 1.000.000 Tokens |

# %% [markdown]
#
# ## Was bedeutet das praktisch?
#
# - **4.000 Tokens** ≈ 3.000 Wörter ≈ 6 Seiten Text
# - **128.000 Tokens** ≈ 96.000 Wörter ≈ Ein kurzes Buch!
# - **1.000.000 Tokens** ≈ 750.000 Wörter ≈ Mehrere Bücher
#
# **Praktische Bedeutung:**
# - Längere Gespräche möglich
# - Ganze Dokumente analysieren
# - Mehr Kontext = bessere Antworten

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

# %% [markdown]
#
# ## Dolma 3 Trainingsdaten (Beispiel)
#
# - Trainingsdaten von Olmo 3 (Allen AI)
# - 9,3 Billionen Token
# - 37 Terabytes Rohtext
# - Als Bücher: 62 Millionen Bücher
#   - Gestapelt: 1.240 km hoch (3x zur ISS)
#   - Vergleich: 140x Mount Everest (8,8 km hoch)
# - Reading time: 53.000 Jahre (24/7 lesen)

# %%
plot_training_data_sources()


# %% [markdown]
#
# ## Pre-Training vs. Fine-Tuning
#
# <br><br>
# <img src="img/llm-pretraining.jpg" alt="Pre-Training vs. Fine-Tuning" style="width: 49%;float: left;margin-right: 2%;">
# <img src="img/llm-sft.jpg" alt="Pre-Training vs. Fine-Tuning" style="width: 49%;">


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

# %% [markdown]
#
# ## Emergente Fähigkeiten (?)
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
plot_emergent_abilities()

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
# ## Halluzinationen: Die Zahlen
#
# **Studien zeigen:**
#
# - **~20%** der von LLMs vorgeschlagenen Software-Pakete existieren nicht!
# - **~48%** des von LLMs generierten Codes enthält Sicherheitslücken
# - LLMs klingen **sehr überzeugend**, auch wenn sie falsch liegen
#
# **Warum?** LLMs sagen wahrscheinliche Muster vorher, nicht Fakten.
# "requests-auth-helper" klingt wie ein echtes Paket - ist es aber nicht!
#
# **Regel**: Immer verifizieren, nie blind vertrauen!

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
# ## Workshop: LLMs erkunden
#
# **Aufgaben** (nutzen Sie ChatGPT, Claude oder ein anderes LLM):
#
# 1. **Verschiedene Prompts testen**: Stellen Sie die gleiche Frage
#    auf 3 verschiedene Arten - beobachten Sie die Unterschiede
#
# 2. **Token-Generierung beobachten**: Aktivieren Sie "Streaming" und
#    beobachten Sie, wie der Text Wort für Wort erscheint
#
# 3. **Wissens-Cutoff testen**: Fragen Sie nach aktuellen Ereignissen
#    (z.B. "Wer hat die Bundestagswahl 2025 gewonnen?")
#
# 4. **Halluzination provozieren**: Fragen Sie nach einem erfundenen
#    Buch oder Paket (z.B. "Erklären Sie das Python-Paket 'super-data-magic'")

# %% [markdown]
#
# ## Workshop: Tipps
#
# **Für Aufgabe 1 (Verschiedene Prompts):**
# - Kurz: "Was ist Python?"
# - Mittel: "Erklären Sie Python für Anfänger"
# - Lang: "Sie sind ein erfahrener Lehrer. Erklären Sie einem Anfänger,
#   was Python ist und warum es beliebt ist. Maximal 3 Sätze."
#
# **Für Aufgabe 4 (Halluzination):**
# - Beobachten Sie, wie überzeugend die Antwort klingt
# - Überprüfen Sie: Existiert das Paket wirklich? (`pip search` oder Google)
# - Das ist ein wichtiger Grund, warum wir LLM-Output immer prüfen!

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - **Prompt Engineering**: Die Kunst, gute Prompts zu schreiben
# - Techniken für bessere LLM-Antworten
# - Praktische Tipps und Muster

# %%
