# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Prompt Engineering</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was ist Prompt Engineering?
#
# - **Prompt**: Die Eingabe, die wir dem LLM geben
# - **Engineering**: Kunst & Wissenschaft, gute Prompts zu schreiben
# - Ziel: Beste Antworten vom LLM bekommen
# - **Wichtig**: Gleiche Frage, unterschiedlich formuliert → unterschiedliche Antworten!

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prompt_engineering_azav_plots import (plot_prompt_quality_spectrum, plot_shot_learning_comparison,
                        plot_chain_of_thought, plot_role_examples,
                        plot_prompt_template)

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Beispiel: Schlechter vs. Guter Prompt
#
# **Schlecht:**
# > "Erkläre Machine Learning"
#
# **Besser:**
# > "Erkläre Machine Learning für einen 10-Jährigen in 3 Sätzen"
#
# **Noch besser:**
# > "Du bist ein geduldiger Lehrer. Erkläre einem 10-jährigen Kind, was Machine Learning ist. Verwende ein Beispiel aus dem Alltag und halte dich auf 3 Sätze."

# %% [markdown]
#
# ## Grundprinzipien guter Prompts
#
# 1. **Sei spezifisch**: Was genau willst du?
# 2. **Gib Kontext**: Wer ist die Zielgruppe?
# 3. **Gib Format vor**: Liste? Absätze? Code?
# 4. **Gib Beispiele**: Few-Shot Learning
# 5. **Iteriere**: Verfeinere den Prompt basierend auf Antworten

# %%

# %% [markdown]
#
# ## Technik 1: Zero-Shot vs. Few-Shot
#
# **Zero-Shot**: Keine Beispiele
# > "Übersetze ins Französische: Hello"
#
# **Few-Shot**: Mit Beispielen
# > "Übersetze ins Französische:\n> Hello → Bonjour\n> Goodbye → Au revoir\n> Thank you → ?"

# %%

# %% [markdown]
#
# ## Technik 2: Chain-of-Thought
#
# - **Problem**: LLMs machen manchmal Fehler bei Logik/Mathe
# - **Lösung**: Bitte um Schritt-für-Schritt-Erklärung
# - "Let's think step by step"
#
# **Ohne CoT:**
# > "Was ist 127 * 43?"
#
# **Mit CoT:**
# > "Was ist 127 * 43? Zeige deine Rechenschritte."

# %%

# %% [markdown]
#
# ## Technik 3: Rollen zuweisen
#
# - LLMs können verschiedene "Rollen" einnehmen
# - Beeinflusst Stil und Qualität der Antwort
#
# **Beispiele:**
# - "Du bist ein erfahrener Python-Programmierer..."
# - "Als geduldiger Lehrer für Kinder..."
# - "In der Rolle eines Wissenschafts-Journalisten..."

# %%

# %% [markdown]
#
# ## Häufige Fehler vermeiden
#
# ❌ **Zu vage**: "Schreibe etwas über KI"
# ✓ **Spezifisch**: "Schreibe einen 200-Wort-Blogpost über KI-Ethik für Laien"
#
# ❌ **Mehrere Fragen**: "Erkläre X und Y und schreibe auch Code"
# ✓ **Eine Sache**: "Erkläre X" (dann separat nach Code fragen)
#
# ❌ **Keine Formatangabe**: Unklare Struktur
# ✓ **Format vorgeben**: "Antworte als nummerierte Liste"

# %% [markdown]
#
# ## Template für gute Prompts
#
# ```
# [Rolle]: Du bist ein [Experte/Lehrer/etc.]
#
# [Aufgabe]: [Genaue Beschreibung der Aufgabe]
#
# [Kontext]: [Zielgruppe, Zweck]
#
# [Format]: [Wie soll die Antwort strukturiert sein]
#
# [Beispiele]: [Optional: 1-3 Beispiele]
#
# [Einschränkungen]: [Länge, Stil, was zu vermeiden ist]
# ```

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Prompt Engineering**: Kunst, LLMs optimal zu nutzen
# - **Spezifisch sein**: Je genauer, desto besser
# - **Techniken**:
#   - Few-Shot Learning (Beispiele geben)
#   - Chain-of-Thought (Schritt-für-Schritt)
#   - Rollen zuweisen
# - **Iterieren**: Prompts basierend auf Antworten verfeinern
# - **Template verwenden**: Strukturierte Prompts sind besser

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - **Praktische Anwendungen** von LLMs
# - Wie man LLMs in eigene Projekte integriert
# - APIs und Tools

# %%
