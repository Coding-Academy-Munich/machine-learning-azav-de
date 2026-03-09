# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Retrieval-Augmented Generation (RAG)</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Das Problem mit LLMs
#
# - LLMs sind sehr leistungsfähig
# - Aber sie haben auch Probleme:
#   - **Halluzinationen**: Sie "erfinden" manchmal Fakten
#   - **Fehlendes Wissen**: Wissen ist auf Trainingsdaten begrenzt
# - Besonders bei spezifischem Wissen (z.B. Firmendaten)

# %%
import os
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import ChatOpenAI

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
)

# %% [markdown]
#
# ## Beispiel: Halluzination / Fehlendes Wissen

# %% [markdown]
#
# Spezifische Frage zum Kursinhalt

# %%

# %%

# %% [markdown]
#
# ## Was ist RAG?
#
# **Retrieval-Augmented Generation** = Abrufen + Generieren
#
# 1. **Retrieval**: Relevante Dokumente/Informationen finden
# 2. **Augmentation**: Kontext zum Prompt hinzufügen
# 3. **Generation**: LLM antwortet basierend auf echten Daten
#
# **Analogie**: Prüfung mit offenem Buch statt aus dem Gedächtnis

# %% [markdown]
#
# ## RAG-Architektur
#
# ```
# Benutzer-Frage
#     ↓
# 1. Relevante Dokumente suchen
#     ↓
# 2. Kontext zum Prompt hinzufügen
#     ↓
# 3. LLM mit Kontext aufrufen
#     ↓
# Antwort basierend auf echten Daten
# ```

# %% [markdown]
#
# ## Beispiel: Mit vs. Ohne Kontext
#
# Beispiel-Dokument als Kontext:

# %%
context = """\
In diesem Kurs lernen Sie über lineare Regression in Kapitel 3.
Lineare Regression ist eine Methode, um lineare Beziehungen zu modellieren.
Wir verwenden die Normalgleichung und den Mean Squared Error.
"""

# %% [markdown]
#
# ### Ohne Kontext
#
# Wie wir gesehen haben, kann das LLM ohne zusätzlichen Kontext die Antwort
# nicht kennen:
#
# ```python
# response_no_context = llm.invoke("Was steht in Kapitel 3 über lineare Regression?")
# ```
#
# **Antwort:**
# ```text
# Da ich keine direkten Websuche- oder Dateizugriffsfunktionen habe, kann ich
# den Inhalt von **Kapitel 3 deines Kurses zur linearen Regression** nicht
# direkt aus einem Kursmaterial (z. B. PDF, Online-Dokument oder Lernplattform)
# für dich herausfinden.
# ```

# %% [markdown]
#
# ### Mit Kontext
#
# Jetzt fügen wir den Kontext hinzu:

# %%
prompt_with_context = f"""\
Basiere deine Antwort NUR auf folgenden Informationen:

{context}

Frage: Was steht in Kapitel 3 über lineare Regression?
"""

# %%
response_with_context = llm.invoke(prompt_with_context)

# %%

# %% [markdown]
#
# ## Wie finden wir den passenden Kontext?
#
# - Wir haben viele Dokumente
# - Welche Granularität der Dokumente sollen wir verwenden?
#   - Vollständige Dokumente?
#   - Absätze?
#   - Sätze?
# - Wie finden wir die relevanten Passagen?<br>
#   Naheliegende Ansätze:
#   - Annotation mit Schlüsselwörtern
#   - Volltextsuche (aus dem Prompt)

# %% [markdown]
#
# ## Probleme bei naheliegenden Ansätzen
#
# - **Schlüsselwörter**:
#   - Müssen manuell gepflegt werden (oder automatisch: fehleranfällig)
#   - Ignorieren viele Aspekte des Textes (bei längeren Dokumenten)
# - **Volltextsuche und Schlüsselwörter**:
#   - Abhängig von exakten Begriffen
#   - Ignorieren semantische Ähnlichkeit

# %% [markdown]
#
# ## Vektor-Suche
#
# <div style="float:right;width:40%;">
#   <img src="img/cosine-distance.png" style="float:right;width:50%"/>
#   <img src="img/vector-difference.png" style="float:left;width:50%"/>
# </div>
# <div style="width:60%;">
# <br>
# <ul>
#   <li>Embeddings: Repräsentation von Text als Vektoren</li>
#   <li>Ähnlichkeit von Vektoren mathematisch definierbar
#   <ul>
#     <li>Beispiel: $|\vec a - \vec b|$ (Euklidische Distanz)</li>
#     <li>Oft besser: Kosinus-Ähnlichkeit<br>(Winkel zwischen Vektoren)</li>
#   </ul>
#   </li>
#   <li>Idee: "ähnliche" Texte → "ähnliche" Vektoren</li>
#   <li>Semantische Suche: Finde ähnliche Vektoren</li>
# </ul>
# </div>

# %% [markdown]
#
# ## RAG-Komponenten
#
# 1. **Dokument-Speicher**: Ihre Dokumente/Daten
# 2. **Embedding-Modell**: Text → Vektoren
# 3. **Vektor-Datenbank**: Für schnelle Suche
# 4. **Retriever**: Findet relevante Dokumente
# 5. **LLM**: Generiert Antwort mit Kontext
#
# **LangChain** bringt alle Komponenten zusammen!

# %% [markdown]
#
# ## Anwendungsfälle für RAG
#
# - **Dokumenten-Q&A**: Fragen zu Handbüchern, Verträgen, etc.
# - **Kundenservice**: Infos aus Wissensdatenbank
# - **Forschungsassistent**: Durchsucht Papers/Artikel
# - **Code-Dokumentation**: Hilfe für Code-Repositories
# - **Firmen-Chatbot**: Zugriff auf interne Dokumente
#
# **Immer wenn**: Spezifisches Wissen benötigt wird!

# %% [markdown]
#
# ## Alternative: Fine-Tuning
#
# - **Fine-Tuning** = Modell mit eigenen Daten weitertrainieren
# - Das Modell "lernt" neue Muster und Verhaltensweisen
# - Beispiele:
#   - Bestimmter Schreibstil (formell, technisch, etc.)
#   - Domänenspezifische Fachsprache
#   - Spezielle Ausgabeformate
# - **Wichtig**: Wissen wird Teil des Modells selbst

# %% [markdown]
#
# ## RAG vs. Fine-Tuning: Vergleich
#
# | Aspekt | RAG | Fine-Tuning |
# |--------|-----|-------------|
# | **Stärke** | Faktenwissen | Verhalten/Stil |
# | **Aktualisierung** | Dokumente austauschen | Neu trainieren |
# | **Kosten** | Gering | Höher |
# | **Transparenz** | Quellen sichtbar | Black Box |
# | **Konsistenz** | Variiert mit Retrieval | Sehr konsistent |
# | **Latenz** | Retrieval-Overhead | Schneller |

# %% [markdown]
#
# ## Wann welchen Ansatz wählen?
#
# <div style="width:50%;float:left;">
# <br>
# <b>RAG ist besser für:</b>
# <ul>
#   <li>Aktuelle Fakten und Dokumente</li>
#   <li>Häufig wechselnde Informationen</li>
#   <li>Nachvollziehbarkeit wichtig (Quellen)</li>
#   <li>Firmen-/Kundendaten</li>
# </ul>
# </div>
# <div style="width:45%;float:left;">
# <br>
# <b>Fine-Tuning ist besser für:</b>
# <ul>
#   <li>Neuer Schreibstil oder Ton</li>
#   <li>Domänenspezifische Sprache</li>
#   <li>Konsistente Ausgabeformate</li>
#   <li>Wenn Retrieval-Latenz problematisch ist</li>
# </ul>
# </div>
# <div style="clear:both;"></div>

# %% [markdown]
#
# ## Herausforderungen ohne Framework
#
# - Dokumente laden und in Stücke aufteilen (Chunking)
# - Embeddings manuell erstellen und verwalten
# - Vektor-Datenbank einrichten und konfigurieren
# - Retrieval-Logik selbst programmieren
# - LLM-Integration von Hand schreiben
#
# **Ergebnis**: ~200-300 Zeilen komplexer Code

# %% [markdown]
#
# ## Wie LangChain hilft
#
# - Fertige Bausteine für jeden RAG-Schritt
# - Komponenten einfach zusammenstecken
# - Viele Datenquellen und Vektor-Datenbanken unterstützt
# - Ergebnis: ~30-50 Zeilen Code
#
# **Darum nutzen wir LangChain von Anfang an!**

# %% [markdown]
#
# ## Was wir lernen werden
#
# 1. **Dokumente laden und aufteilen**: Document Loaders und Text Chunking
# 2. **Vector Embeddings**: Wie Text als Vektoren dargestellt wird
# 3. **Vektor-Datenbank (Qdrant)**: Speichern und Suchen mit Hybrid Search
# 4. **RAG mit LangChain**: Alles zusammenbauen
# 5. **Workshop**: Eigenes RAG-System mit Gradio-UI
#
# **Ziel**: Ein funktionierendes Q&A-System über Ihre eigenen Dokumente!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Problem**: LLMs halluzinieren bei spezifischem Wissen
# - **Lösung**: RAG - Abrufen + Generieren
# - **Wie**: Relevante Dokumente finden und als Kontext nutzen
# - **Vorteil**: Präzise Antworten mit echten Daten
# - **LangChain**: Macht RAG einfach
#
# **Nächster Schritt**: Dokumente laden mit Document Loaders!

# %% [markdown]
#
# ## Workshop: RAG vs. Fine-Tuning
#
# Entscheiden Sie für jedes Szenario, ob **RAG** oder **Fine-Tuning** besser
# geeignet ist, und begründen Sie Ihre Wahl kurz.

# %% [markdown]
#
# **Szenario 1:** Ein Krankenhaus möchte einen Chatbot erstellen, der Ärzten
# Informationen aus den neuesten medizinischen Leitlinien gibt. Die Leitlinien
# werden regelmäßig aktualisiert.

# %% [markdown]
# *Antwort:* 

# %% [markdown]
#
# **Szenario 2:** Ein Unternehmen möchte, dass ein LLM alle E-Mails in einem
# bestimmten formellen Stil schreibt, der zur Markenidentität passt.

# %% [markdown]
# *Antwort:* 

# %% [markdown]
#
# **Szenario 3:** Eine Anwaltskanzlei möchte einen Assistenten, der Fragen zu
# den Verträgen und internen Dokumenten der Kanzlei beantworten kann.

# %% [markdown]
# *Antwort:* 

# %% [markdown]
#
# **Szenario 4:** Ein Medizintechnik-Unternehmen möchte, dass ein LLM
# Befundberichte in einem spezifischen medizinischen Fachformat ausgibt, das
# dem klinikinternen Standard entspricht.

# %% [markdown]
# *Antwort:* 
