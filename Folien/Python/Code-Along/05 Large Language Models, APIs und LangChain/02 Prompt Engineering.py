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
# ## Anatomie eines guten Prompts
#
# Jeder gute Prompt enthält diese **4 Kern-Elemente**:
#
# 1. **Kontext**: Was ist die Situation?
# 2. **Anforderung**: Was genau soll das LLM tun?
# 3. **Einschränkungen**: Format, Länge, Stil
# 4. **Beispiele**: Optional, aber sehr hilfreich (Few-Shot)

# %% [markdown]
#
# ## Beispiel: Anatomie anwenden
#
# ```
# [Kontext]
# Ich entwickle eine Python-Anwendung für Dateimanagement.
#
# [Anforderung]
# Schreibe eine Funktion, die alle Dateien in einem Ordner auflistet.
#
# [Einschränkungen]
# - Verwende pathlib (nicht os)
# - Füge Type Hints hinzu
# - Maximal 10 Zeilen
#
# [Beispiel-Output]
# def list_files(folder: Path) -> list[Path]:
#     ...
# ```

# %% [markdown]
#
# ## Grundprinzipien guter Prompts
#
# 1. **Seien Sie spezifisch**: Was genau wollen Sie?
# 2. **Geben Sie Kontext**: Wer ist die Zielgruppe?
# 3. **Geben Sie Format vor**: Liste? Absätze? Code?
# 4. **Geben Sie Beispiele**: Few-Shot Learning
# 5. **Iterieren Sie**: Verfeinern Sie den Prompt basierend auf Antworten

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
# ## Praktische Prompt-Muster
#
# **4 häufige Muster für Programmieraufgaben:**
#
# 1. **Generieren**: "Schreibe eine Funktion, die..."
# 2. **Verbessern**: "Refaktoriere diesen Code, um..."
# 3. **Erweitern**: "Füge Fehlerbehandlung hinzu für..."
# 4. **Erklären**: "Erkläre diesen Code..."

# %% [markdown]
#
# ## Muster in Aktion
#
# **Generieren:**
# > "Schreibe eine Python-Funktion, die prüft, ob eine Zahl eine Primzahl ist. Verwende Type Hints."
#
# **Verbessern:**
# > "Refaktoriere diese Funktion, um List Comprehensions statt Schleifen zu verwenden."
#
# **Erweitern:**
# > "Füge Validierung hinzu: Die Funktion soll einen ValueError werfen, wenn die Eingabe negativ ist."
#
# **Erklären:**
# > "Erkläre Zeile für Zeile, was dieser Code macht."

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
# ## Iteratives Verfeinern
#
# **Prompts werden selten beim ersten Mal perfekt!**
#
# Der Workflow:
# 1. **Prompt schreiben** → Antwort erhalten
# 2. **Antwort bewerten** → Was fehlt? Was ist falsch?
# 3. **Prompt verfeinern** → Spezifischer, mehr Kontext
# 4. **Wiederholen** → Bis zufrieden (max. 3-4 Iterationen)
#
# **Wichtig**: Nach 3-4 Iterationen ohne Verbesserung → anderen Ansatz versuchen!

# %% [markdown]
#
# ## Effektives Feedback geben
#
# **Vages Feedback** (❌):
# > "Das ist nicht gut genug"
#
# **Spezifisches Feedback** (✓):
# > "Die Funktion behandelt keine leeren Listen. Füge eine Prüfung hinzu, die einen leeren String zurückgibt."
#
# **Noch besser** (✓✓):
# > "Problem: Die Funktion gibt None zurück bei leerer Liste.
# > Erwartetes Verhalten: Leerer String zurückgeben.
# > Bitte korrigiere die Funktion entsprechend."

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
# ## Workshop: Prompt Engineering Praxis
#
# **Aufgaben** (nutzen Sie ChatGPT, Claude oder ein anderes LLM):
#
# 1. **Bad → Good**: Nehmen Sie den Prompt "Erkläre Python" und verbessern Sie ihn
#    mit allen 4 Kernelementen
#
# 2. **Few-Shot**: Erstellen Sie einen Prompt mit 2-3 Beispielen, der
#    Datumsformate konvertiert (z.B. "15.03.2024" → "March 15, 2024")
#
# 3. **Chain-of-Thought**: Lassen Sie das LLM ein Logik-Rätsel lösen und
#    dabei seine Schritte erklären
#
# 4. **Iterieren**: Schreiben Sie einen Prompt für eine Funktion, dann
#    verbessern Sie ihn 3x basierend auf den Ergebnissen

# %% [markdown]
#
# ## Workshop: Beispiellösung Aufgabe 1
#
# **Vorher:**
# > "Erkläre Python"
#
# **Nachher:**
# > "[Kontext] Ich bin Anfänger und möchte programmieren lernen.
# >
# > [Anforderung] Erkläre mir, was Python ist und warum es eine gute
# > erste Programmiersprache ist.
# >
# > [Format] 3 kurze Absätze mit Überschriften.
# >
# > [Einschränkungen] Vermeide Fachbegriffe ohne Erklärung."

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - **LLM APIs**: LLMs programmatisch nutzen
# - Wie man LLMs in eigene Anwendungen integriert
# - OpenAI, Anthropic und andere Anbieter

# %%
