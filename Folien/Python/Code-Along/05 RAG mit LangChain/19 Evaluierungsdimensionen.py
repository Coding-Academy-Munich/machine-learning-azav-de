# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Evaluierungsdimensionen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was genau evaluieren wir?
#
# Ein RAG-System hat **zwei Hauptkomponenten**:
#
# 1. **Retrieval**: Dokumente finden
# 2. **Generation**: Antwort erzeugen
#
# Beide können unabhängig voneinander versagen!

# %% [markdown]
#
# ## Dimension 1: Retrieval-Qualität
#
# **"Haben wir die richtigen Dokumente gefunden?"**
#
# - **Context Precision** (Kontext-Präzision):
#   - Von 4 abgerufenen Dokumenten — wie viele waren nützlich?
#   - 4 nützliche von 4 abgerufenen → Precision = 100%
#   - 1 nützliches von 4 abgerufenen → Precision = 25%
# - **Context Recall** (Kontext-Abdeckung):
#   - Haben wir alle nützlichen Dokumente gefunden?
#   - 3 von 3 nützlichen gefunden → Recall = 100%
#   - 1 von 3 nützlichen gefunden → Recall = 33%

# %% [markdown]
#
# ## Dimension 2: Generierungsqualität
#
# **"Ist die Antwort gut?"**
#
# - **Faithfulness** (Treue zum Kontext):
#   - Basiert die Antwort nur auf den abgerufenen Dokumenten?
#   - Oder hat das LLM Informationen "erfunden"?
#   - Wie bei einer Prüfung: Nur aus den Unterlagen antworten!
# - **Answer Relevancy** (Antwort-Relevanz):
#   - Beantwortet die Antwort tatsächlich die gestellte Frage?
#   - Oder geht sie am Thema vorbei?

# %% [markdown]
#
# ## Beispiel: Gute vs. schlechte Antworten
#
# **Frage**: "Was ist Overfitting?"
#
# **Kontext**: "Overfitting tritt auf, wenn ein Modell die Trainingsdaten
# auswendig lernt. Regularisierung hilft dagegen."
#
# | Antwort | Faithful? | Relevant? |
# |---------|-----------|-----------|
# | "Overfitting = Modell lernt Trainingsdaten auswendig" | Ja | Ja |
# | "Overfitting ist wie wenn man nur für eine Prüfung lernt" | Nein (eigenes Wissen) | Ja |
# | "Regularisierung hilft gegen Overfitting" | Ja | Teilweise |
# | "Python ist eine Programmiersprache" | Nein | Nein |

# %% [markdown]
#
# ## Die vier Metriken im Überblick
#
# | Metrik | Frage | Braucht |
# |--------|-------|---------|
# | **Context Precision** | Waren die gefundenen Docs relevant? | Frage + Kontext |
# | **Context Recall** | Haben wir genug gefunden? | Kontext + Referenz-Antwort |
# | **Faithfulness** | Basiert die Antwort auf dem Kontext? | Antwort + Kontext |
# | **Answer Relevancy** | Beantwortet es die Frage? | Frage + Antwort |
#
# Diese vier Metriken decken die wichtigsten Qualitätsdimensionen ab

# %% [markdown]
#
# ## Was brauchen wir für die Evaluierung?
#
# Für jede Test-Frage sammeln wir:
#
# 1. **Frage** (user_input): Was der Benutzer fragt
# 2. **Abgerufene Dokumente** (retrieved_contexts): Was der Retriever findet
# 3. **Generierte Antwort** (response): Was das LLM antwortet
# 4. **Referenz-Antwort** (reference): Was die "richtige" Antwort wäre
#
# Mit diesen vier Elementen können wir alle Metriken berechnen

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Retrieval-Qualität**: Richtige Dokumente gefunden? (Precision, Recall)
# - **Generierungsqualität**: Gute Antwort erzeugt? (Faithfulness, Relevancy)
# - Vier Schlüsselmetriken decken beide Dimensionen ab
# - Für jede Evaluierung brauchen wir: Frage, Kontext, Antwort, Referenz
#
# **Nächster Schritt**: Manuell evaluieren, bevor wir automatisieren!
