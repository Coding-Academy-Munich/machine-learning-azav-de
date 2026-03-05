# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Deep Dive: LLM-als-Richter</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Wie funktioniert LLM-als-Richter?
#
# RAGAS und LangSmith nutzen LLMs, um Antworten zu bewerten.
#
# **Das allgemeine Muster:**
# 1. Eingabe in einzelne **Aussagen** zerlegen
# 2. Jede Aussage gegen die **Evidenz** prüfen
# 3. Einzelbewertungen zu einem **Score** zusammenfassen
#
# In diesem Deep Dive schauen wir unter die Haube.

# %% [markdown]
#
# ## RAGAS Faithfulness: Schritt für Schritt
#
# **Schritt 1: Aussagen extrahieren**
#
# Das LLM bekommt die generierte Antwort und soll einzelne Aussagen
# (Claims) extrahieren:
#
# ```
# Antwort: "Neural Networks werden durch Backpropagation trainiert.
#           Das Training dauert typischerweise Stunden auf einer GPU."
#
# Extrahierte Aussagen:
# 1. "Neural Networks werden durch Backpropagation trainiert"
# 2. "Das Training dauert typischerweise Stunden auf einer GPU"
# ```

# %% [markdown]
#
# ## Faithfulness: Schritt 2
#
# **Schritt 2: Natural Language Inference (NLI)**
#
# Für jede Aussage fragt das LLM: "Wird diese Aussage durch den
# Kontext gestützt?"
#
# ```
# Kontext: "Neural Networks bestehen aus Schichten von Neuronen.
#           Training erfolgt durch Backpropagation und Gradient Descent."
#
# Aussage 1: "NNs werden durch Backpropagation trainiert"
# → Urteil: GESTÜTZT (steht im Kontext)
#
# Aussage 2: "Training dauert Stunden auf GPU"
# → Urteil: NICHT GESTÜTZT (nicht im Kontext)
# ```

# %% [markdown]
#
# ## Faithfulness: Schritt 3
#
# **Schritt 3: Aggregation**
#
# $$\text{Faithfulness} = \frac{|\text{gestützte Aussagen}|}{|\text{alle Aussagen}|}$$
#
# In unserem Beispiel:
#
# $$\text{Faithfulness} = \frac{1}{2} = 0.5$$
#
# **Interpretation**: 50% der Antwort basiert auf dem Kontext

# %% [markdown]
#
# ## Der vereinfachte Prompt
#
# So ähnlich sieht der Prompt aus, den RAGAS intern verwendet
# (vereinfacht):
#
# ```
# Given the context and a claim, determine if the claim
# can be inferred from the context.
#
# Context: {context}
# Claim: {claim}
#
# Classify as:
# - "supported": The claim is directly supported by the context
# - "not_supported": The claim cannot be inferred from the context
# ```
#
# RAGAS macht das für **jede** extrahierte Aussage einzeln

# %% [markdown]
#
# ## Response Relevancy: Wie es funktioniert
#
# Ein anderer Ansatz — generiert **Fragen aus der Antwort**:
#
# 1. **Fragen generieren**: Das LLM erstellt Fragen, die die Antwort beantworten würde
# 2. **Ähnlichkeit berechnen**: Wie ähnlich sind die generierten Fragen zur Original-Frage?
# 3. **Score**: Durchschnittliche Ähnlichkeit
#
# ```
# Original-Frage: "Was ist Overfitting?"
#
# Generierte Fragen aus der Antwort:
# - "Was passiert, wenn ein Modell Trainingsdaten auswendig lernt?" (ähnlich!)
# - "Welche Techniken gibt es gegen Overfitting?" (teilweise ähnlich)
# ```

# %% [markdown]
#
# ## Bekannte Biases bei LLM-als-Richter
#
# LLMs als Richter sind nicht perfekt. Bekannte Verzerrungen:
#
# - **Position Bias**: Das LLM bevorzugt die erste oder letzte Option
# - **Verbosity Bias**: Längere Antworten werden als besser bewertet
# - **Self-Enhancement Bias**: Ein LLM bewertet Antworten seines eigenen
#   Modells besser
# - **Format Bias**: Gut formatierte Antworten (Listen, Überschriften)
#   werden bevorzugt

# %% [markdown]
#
# ## Best Practices
#
# - **Verschiedene Modelle**: Evaluator-LLM ≠ RAG-LLM
#   - Reduziert Self-Enhancement Bias
# - **Mehrfache Bewertung**: Gleiche Aussage mehrmals bewerten lassen
#   - Reduziert Zufallsvariation
# - **Klare Prompts**: Präzise Bewertungskriterien definieren
# - **Manuelles Stichproben**: Stichprobenartig manuelle Bewertung zum Vergleich
# - **Kalibrierung**: Bekannte Beispiele nutzen, um den Richter zu kalibrieren

# %% [markdown]
#
# ## Kosten-Analyse
#
# Für eine einzelne Faithfulness-Bewertung:
#
# | Schritt | LLM-Aufrufe |
# |---------|------------|
# | Aussagen extrahieren | 1 |
# | Jede Aussage prüfen (z.B. 4 Aussagen) | 4 |
# | **Gesamt pro Beispiel** | **5** |
#
# Bei 3 Metriken und 10 Beispielen:
# - Faithfulness: ~50 Aufrufe
# - Response Relevancy: ~30 Aufrufe
# - Context Recall: ~30 Aufrufe
# - **Gesamt: ~110 LLM-Aufrufe**
#
# Mit ministral-14b (~$0.001/Aufruf): **~$0.11** pro Evaluierungslauf

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Faithfulness** funktioniert in 3 Schritten:
#   Aussagen extrahieren → NLI-Prüfung → Aggregation
# - **Response Relevancy** generiert Fragen aus der Antwort und
#   vergleicht sie mit der Original-Frage
# - **LLM-als-Richter hat Biases**: Position, Verbosity, Self-Enhancement
# - **Best Practices**: Verschiedene Modelle, klare Prompts, manuelle Stichproben
# - **Kosten**: ~100 LLM-Aufrufe pro Evaluierungslauf (sehr günstig)
