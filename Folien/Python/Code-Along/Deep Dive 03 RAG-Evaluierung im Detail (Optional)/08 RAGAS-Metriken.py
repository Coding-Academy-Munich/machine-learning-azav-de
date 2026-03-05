# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>RAGAS-Metriken</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Die drei wichtigsten RAGAS-Metriken
#
# | Metrik | Prüft | Score |
# |--------|-------|-------|
# | **Faithfulness** | Basiert die Antwort auf dem Kontext? | 0-1 |
# | **Response Relevancy** | Beantwortet es die Frage? | 0-1 |
# | **Context Recall** | Wurden genug Infos gefunden? | 0-1 |
#
# Jede Metrik gibt einen Wert zwischen 0 (schlecht) und 1 (perfekt)

# %% [markdown]
#
# ## Faithfulness (Treue zum Kontext)
#
# **"Basiert die Antwort nur auf den abgerufenen Dokumenten?"**
#
# Analogie: Wie bei einer **Open-Book-Prüfung**
# - Erlaubt: Informationen aus den Unterlagen verwenden
# - Nicht erlaubt: Eigenes Wissen einbringen
#
# **Wie es funktioniert:**
# 1. RAGAS extrahiert einzelne **Aussagen** aus der Antwort
# 2. Für jede Aussage: Steht das im Kontext?
# 3. Score = Anteil der gestützten Aussagen
#
# **Beispiel**: 3 von 4 Aussagen im Kontext → Faithfulness = 0.75

# %% [markdown]
#
# ## Faithfulness: Konkretes Beispiel
#
# **Kontext**: "Neural Networks werden durch Backpropagation trainiert."
#
# **Antwort**: "Neural Networks werden durch Backpropagation trainiert.
# Das Training dauert typischerweise mehrere Stunden auf einer GPU."
#
# | Aussage | Im Kontext? |
# |---------|------------|
# | "NNs werden durch Backpropagation trainiert" | Ja |
# | "Training dauert mehrere Stunden auf GPU" | **Nein** (Halluzination!) |
#
# **Faithfulness = 1/2 = 0.5**

# %% [markdown]
#
# ## Response Relevancy (Antwort-Relevanz)
#
# **"Beantwortet die Antwort tatsächlich die gestellte Frage?"**
#
# Analogie: Hat der Schüler die **richtige Frage** beantwortet?
#
# **Wie es funktioniert:**
# 1. RAGAS generiert mögliche Fragen **aus der Antwort**
# 2. Vergleicht diese mit der **originalen Frage**
# 3. Je ähnlicher, desto relevanter
#
# Eine Antwort kann korrekt sein, aber trotzdem irrelevant!

# %% [markdown]
#
# ## Response Relevancy: Konkretes Beispiel
#
# **Frage**: "Was ist Overfitting?"
#
# | Antwort | Relevancy |
# |---------|-----------|
# | "Overfitting = Modell lernt Trainingsdaten auswendig" | **Hoch** |
# | "Es gibt viele ML-Methoden: Regression, NNs, Trees..." | **Niedrig** |
# | "Regularisierung hilft gegen viele Probleme" | **Mittel** |
#
# Die zweite Antwort ist zwar über ML, aber beantwortet die Frage nicht

# %% [markdown]
#
# ## Context Recall (Kontext-Abdeckung)
#
# **"Haben wir genug relevante Informationen abgerufen?"**
#
# Analogie: Hat der Schüler die **richtigen Seiten** aufgeschlagen?
#
# **Wie es funktioniert:**
# 1. RAGAS zerlegt die **Referenz-Antwort** in einzelne Aussagen
# 2. Für jede Aussage: Ist sie in den abgerufenen Dokumenten enthalten?
# 3. Score = Anteil der abgedeckten Aussagen
#
# **Wichtig**: Diese Metrik braucht eine Referenz-Antwort!

# %% [markdown]
#
# ## Context Recall: Konkretes Beispiel
#
# **Referenz-Antwort**: "Overfitting tritt auf, wenn ein Modell die
# Trainingsdaten auswendig lernt. Regularisierung und Cross-Validation helfen."
#
# **Abgerufener Kontext**: "Overfitting tritt auf, wenn ein Modell die
# Trainingsdaten auswendig lernt. Dropout ist eine Technik dagegen."
#
# | Aussage aus Referenz | Im Kontext? |
# |---------------------|------------|
# | "Overfitting = auswendig lernen" | Ja |
# | "Regularisierung hilft" | **Nein** |
# | "Cross-Validation hilft" | **Nein** |
#
# **Context Recall = 1/3 = 0.33** → Retriever hat zu wenig gefunden!

# %% [markdown]
#
# ## Wann welche Metrik?
#
# | Situation | Metrik | Warum |
# |-----------|--------|-------|
# | Halluzinationen prüfen | **Faithfulness** | Erkennt erfundene Infos |
# | Antwortqualität prüfen | **Response Relevancy** | Erkennt Themaverfehlung |
# | Retriever prüfen | **Context Recall** | Erkennt schlechte Suche |
#
# **Empfehlung**: Immer alle drei verwenden!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Faithfulness**: Nur Infos aus dem Kontext? (Halluzinationserkennung)
# - **Response Relevancy**: Frage wirklich beantwortet?
# - **Context Recall**: Genug relevante Dokumente gefunden?
# - Alle Metriken geben Werte von 0 (schlecht) bis 1 (perfekt)
# - Jede Metrik nutzt ein LLM unter der Haube
#
# **Nächster Schritt**: RAGAS auf unser RAG-System anwenden!
