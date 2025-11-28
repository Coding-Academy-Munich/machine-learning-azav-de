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
# - Aber sie haben ein Problem: **Halluzinationen**
# - Sie "erfinden" manchmal Fakten
# - Besonders bei spezifischem Wissen (z.B. Firmendaten)

# %%
import os

# %%
from langchain_openai import ChatOpenAI

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    temperature=0
)

# %% [markdown]
#
# ## Beispiel: Halluzination

# %% [markdown]
#
# Fragen zu etwas sehr Spezifischem

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
# ## RAG-Komponenten
#
# 1. **Dokument-Speicher**: Ihre Dokumente/Daten
# 2. **Vektor-Datenbank**: Für schnelle Suche
# 3. **Embedding-Modell**: Text → Vektoren
# 4. **Retriever**: Findet relevante Dokumente
# 5. **LLM**: Generiert Antwort mit Kontext
#
# **LangChain** bringt alle Komponenten zusammen!

# %% [markdown]
#
# ## Beispiel: Mit vs. Ohne Kontext

# %%
# Context document
context = """
In diesem Kurs lernen Sie über lineare Regression in Kapitel 3.
Lineare Regression ist eine Methode, um lineare Beziehungen zu modellieren.
Wir verwenden die Normalgleichung und den Mean Squared Error.

In this course you learn about linear regression in Chapter 3.
Linear regression is a method to model linear relationships.
We use the normal equation and Mean Squared Error.
"""

# %%

# %%

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
# ## RAG vs. Fine-Tuning
#
# ### RAG:
# - ✅ Einfach: Dokumente hinzufügen
# - ✅ Aktuell: Daten können sich ändern
# - ✅ Günstig: Kein Modell-Training
# - ✅ Transparent: Sehen, woher Info kommt
#
# ### Fine-Tuning:
# - Teuer: Modell neu trainieren
# - Aufwändig: Große Datensätze nötig
# - Statisch: Änderungen = Neu-Training

# %% [markdown]
#
# ## Wie LangChain RAG vereinfacht
#
# **Ohne Framework**:
# - Dokumente laden und chunken
# - Embeddings manuell erstellen
# - Vektor-Datenbank einrichten
# - Retrieval-Logik programmieren
# - LLM-Integration schreiben
# - ~200-300 Zeilen Code
#
# **Mit LangChain**:
# - Komponenten zusammenstecken
# - ~30-50 Zeilen Code
#
# **Darum nutzen wir LangChain von Anfang an!**

# %% [markdown]
#
# ## Was wir lernen werden
#
# 1. **Vector Embeddings**: Wie Text als Vektoren dargestellt wird
# 2. **ChromaDB**: Einfache Vektor-Datenbank
# 3. **RAG mit LangChain**: Alles zusammenbauen
# 4. **Workshop**: Eigenes RAG-System mit Gradio-UI
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
# **Nächster Schritt**: Vector Embeddings verstehen!

# %%
