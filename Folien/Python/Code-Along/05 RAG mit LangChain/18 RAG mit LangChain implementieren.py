# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>RAG mit LangChain implementieren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Was wir bis jetzt haben
#
# - ✅ LLMs nutzen mit LangChain
# - ✅ Text bereinigen und in Chunks aufteilen
# - ✅ Embeddings erstellen (Kosinus-Ähnlichkeit)
# - ✅ Qdrant für Vektor-Speicherung + **Hybrid Search**
#
# **Jetzt**: Alles zusammenbauen zu einem RAG-System!

# %%
# ! pip install qdrant-client langchain-qdrant

# %%
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# %%
load_dotenv()

# %% [markdown]
#
# ## RAG-Pipeline Übersicht
#
# **Einmalig (Setup)**:
# 1. Dokumente laden
# 2. Text bereinigen und in Chunks aufteilen
# 3. In Qdrant speichern (mit Hybrid Search)
#
# **Bei jeder Anfrage**:
# 4. Anfrage embedden
# 5. Ähnlichste Chunks finden (Retrieval)
# 6. Als Kontext an LLM geben (Augmentation)
# 7. Antwort generieren (Generation)

# %% [markdown]
#
# ## Schritt 1: Dokumente laden und chunken

# %%
docs_content = [
    """Linear Regression ist eine Methode des überwachten Lernens.
    Sie modelliert lineare Beziehungen zwischen Features und Zielvariable.
    Die Normalgleichung kann verwendet werden, um optimale Parameter zu finden.""",

    """Neural Networks bestehen aus Schichten von Neuronen.
    Jedes Neuron berechnet eine gewichtete Summe und wendet eine Aktivierungsfunktion an.
    Training erfolgt durch Backpropagation und Gradient Descent.""",

    """Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt.
    Regularisierung und Cross-Validation helfen, Overfitting zu vermeiden.
    Ein gutes Modell generalisiert auf neue, ungesehene Daten.""",

    """Large Language Models sind sehr große neuronale Netze.
    Sie werden auf Milliarden von Wörtern trainiert.
    RAG hilft LLMs, präziser mit spezifischem Wissen zu antworten."""
]

# %%
documents = [Document(page_content=doc) for doc in docs_content]

# %% [markdown]
#
# Unsere Beispiel-Dokumente sind kurz genug, dass wir sie nicht aufteilen müssen.
# In einer echten Anwendung würden wir `RecursiveCharacterTextSplitter` verwenden,
# wie in der Lektion zu Text Processing gelernt.

# %% [markdown]
# Anzahl Dokumente:

# %%

# %% [markdown]
#
# ## Schritt 2: Vektor-Store erstellen

# %%
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")


# %% [markdown]
#
# Erstellen Sie die Embeddings und den Qdrant Vektor-Store aus den Dokumenten.
# Verwenden Sie `RetrievalMode.HYBRID` mit `sparse_embeddings` für Hybrid Search.

# %%

# %%

# %% [markdown]
#
# ## Schritt 3: Retriever erstellen
#
# - Der **Retriever** durchsucht den Vektor-Store
# - Gibt die `k` ähnlichsten Dokumente zurück
# - Wird aus dem Vektor-Store erstellt

# %% [markdown]
#
# Erstellen Sie einen Retriever aus dem Vektor-Store mit `as_retriever()`.

# %%

# %% [markdown]
#
# ## Retriever testen

# %%

# %%

# %% [markdown]
#
# ## Schritt 4: RAG Chain — Warum format_docs?
#
# - Der Retriever gibt **Document-Objekte** zurück
# - Das LLM braucht aber **normalen Text**
# - `format_docs` verbindet alle Dokumente zu einem Text

# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %%

# %%

# %% [markdown]
#
# ## Schritt 4a: Retrieval + Formatierung
#
# Mit LCEL (LangChain Expression Language) können wir Schritte verketten:
#
# ```
# retriever | format_docs
# ```
#
# - `|` bedeutet: "Output von links wird Input von rechts"
# - Frage → Retriever → Dokumente → format_docs → Text

# %%

# %% [markdown]
# Kontext-Text:

# %%
