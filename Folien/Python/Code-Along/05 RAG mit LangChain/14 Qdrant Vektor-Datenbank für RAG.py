# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Qdrant: Vektor-Datenbank für RAG</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Warum Vektor-Datenbanken?
#
# - **Problem**: Viele Dokumente durchsuchen ist langsam
# - **Lösung**: Vektor-Datenbank
# - Speichert Embeddings effizient
# - Findet ähnlichste Vektoren schnell
# - **Millionen** von Dokumenten in Millisekunden!

# %% [markdown]
#
# ## Warum Qdrant?
#
# - ✅ **Open Source** (Apache 2.0 Lizenz)
# - ✅ **Einfach zu installieren**: `pip install qdrant-client langchain-qdrant`
# - ✅ **Funktioniert auf Windows und Linux**
# - ✅ **Keine Server-Konfiguration** nötig (lokaler Modus)
# - ✅ **Hybrid Search**: Semantische + Keyword-Suche kombiniert
# - ✅ **LangChain-Integration** out-of-the-box
#
# **Ideal für RAG-Systeme!**

# %% [markdown]
#
# ## Installation
#
# ```bash
# pip install qdrant-client langchain-qdrant
# ```
#
# **Das war's!** Keine Docker, keine Server-Konfiguration.

# %%
# ! pip install qdrant-client langchain-qdrant

# %%
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# %%
load_dotenv()

# %% [markdown]
#
# ## Qdrant Modi
#
# 1. **In-Memory**: Daten nur im RAM (für Tests)
#    - `location=":memory:"`
# 2. **Persistent**: Daten auf Festplatte (empfohlen)
#    - `path="./qdrant_db"`
#
# **Für RAG**: Persistent nutzen!
#
# Wir verwenden Qdrant über die **LangChain-Integration**

# %% [markdown]
#
# ## Embedding-Modell erstellen
#
# - Qdrant braucht ein **Embedding-Modell** (wir kennen das aus der letzten Lektion)
# - Wir verwenden OpenAI Embeddings über OpenRouter

# %%
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

# %% [markdown]
#
# ## Vektor-Store erstellen und Dokumente hinzufügen
#
# - `QdrantVectorStore.from_texts()` erstellt den Store und fügt Texte hinzu
# - Embeddings werden **automatisch** berechnet

# %%
texts = [
    "Neural networks can learn complex patterns",
    "Gradient descent optimizes model parameters",
    "Overfitting occurs when models memorize training data",
    "RAG combines retrieval with generation",
]

# TODO: Create QdrantVectorStore from texts

# %%

# %% [markdown]
#
# ## Dokumente suchen
#
# - `similarity_search()` findet die ähnlichsten Dokumente
# - Parameter `k` bestimmt die Anzahl der Ergebnisse

# %%
# TODO: Search for "What is overfitting?"

# %%

# %% [markdown]
#
# ## Suche mit Relevanz-Scores
#
# - `similarity_search_with_score()` zeigt auch die Ähnlichkeit
# - **Niedriger Score** = hohe Ähnlichkeit (Distanz)

# %%

# %%

# %% [markdown]
#
# ## Was passiert bei irrelevanten Anfragen?

# %%

# %%

# %% [markdown]
#
# ## Vektor-Datenbanken liefern immer Ergebnisse!
#
# - Auch bei irrelevanten Fragen gibt die Datenbank Ergebnisse zurück
# - Sie wählt einfach die **ähnlichsten** Dokumente — auch wenn keines relevant ist
# - **Hoher Score** (Distanz) = niedrige Relevanz
# - Wir können nach Score filtern, um irrelevante Ergebnisse auszuschließen
# - Das ist wichtig für robuste RAG-Systeme!

# %% [markdown]
#
# ## Metadata verwenden
#
# - Dokumente können **Metadata** haben (z.B. Thema, Datum)
# - Nützlich für zusätzliche Filterung

# %%
from langchain_core.documents import Document

docs_with_meta = [
    Document(page_content="Linear regression models linear relationships", metadata={"topic": "ML"}),
    Document(page_content="Python is a popular programming language", metadata={"topic": "Programming"}),
    Document(page_content="Decision trees split data into branches", metadata={"topic": "ML"}),
]

vectorstore_meta = QdrantVectorStore.from_documents(
    documents=docs_with_meta,
    embedding=embeddings,
    collection_name="docs_with_meta",
    location=":memory:",
)

# %%

# %% [markdown]
#
# ## Retriever erstellen
#
# - Ein **Retriever** ist die Standard-Schnittstelle in LangChain
# - Wird für RAG-Chains verwendet
# - Einfach aus dem Vektor-Store erstellen

# %%
# TODO: Create retriever from vectorstore

# %%

# %% [markdown]
#
# ## Persistenter Speicher
#
# - Für echte Anwendungen: Daten auf Festplatte speichern
# - Qdrant speichert Daten lokal mit `path=`

# %% [markdown]
#
# ```python
# vectorstore = QdrantVectorStore.from_documents(
#     documents=documents,
#     embedding=embeddings,
#     collection_name="my_docs",
#     path="./qdrant_db",  # Daten werden hier gespeichert
# )
# ```

# %% [markdown]
#
# ## Hybrid Search: Das Beste aus beiden Welten
#
# - **Semantische Suche**: Findet inhaltlich ähnliche Dokumente
#   - "Hund" findet auch "Welpe" und "Vierbeiner"
# - **Keyword-Suche (BM25)**: Findet exakte Begriffe
#   - "BM25" findet nur Dokumente mit genau diesem Wort
# - **Hybrid Search**: Kombiniert beides!
#   - Semantisches Verständnis + exakte Keyword-Treffer

# %% [markdown]
#
# ## Hybrid Search mit Qdrant
#
# - Qdrant unterstützt Hybrid Search **nativ**
# - Braucht zwei Arten von Embeddings:
#   - **Dense Embeddings**: Für semantische Suche (wie bisher)
#   - **Sparse Embeddings**: Für Keyword-Suche (BM25)
# - `langchain_qdrant` macht das einfach!

# %%
# ! pip install fastembed-gpu

# %%
from langchain_qdrant import FastEmbedSparse, RetrievalMode

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# %% [markdown]
#
# ## Hybrid-Vektor-Store erstellen

# %%
hybrid_docs = [
    "Neural networks use backpropagation for training",
    "The BM25 algorithm ranks documents by term frequency",
    "Gradient descent minimizes the loss function",
    "Hybrid search combines BM25 keyword matching with semantic similarity",
    "Overfitting happens when a model memorizes training data",
]

# TODO: Create QdrantVectorStore with RetrievalMode.HYBRID

# %%

# %% [markdown]
#
# ## Hybrid Search vs. Semantische Suche
#
# Vergleichen wir die Ergebnisse für eine Keyword-lastige Anfrage:

# %%
query = "BM25"

# %% [markdown]
#
# Nur semantische Suche:

# %%

# %%

# %% [markdown]
#
# Hybrid Search (semantisch + BM25):

# %%

# %%

# %% [markdown]
#
# ## Warum Hybrid Search besser ist
#
# - **Semantische Suche** versteht Bedeutung, kann aber exakte Begriffe verfehlen
# - **BM25** findet exakte Begriffe, versteht aber keine Synonyme
# - **Hybrid** kombiniert beide Stärken:
#   - Findet exakte Fachbegriffe (z.B. "BM25", "LSTM", "MSE")
#   - Versteht auch Umschreibungen (z.B. "neuronales Netz" ≈ "Deep Learning")
# - In der Praxis: **Hybrid Search liefert bessere RAG-Ergebnisse**

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Qdrant**: Open-Source Vektor-Datenbank mit Hybrid Search
# - **Einfach**: Pip-Install, keine Server-Konfiguration
# - **LangChain-Integration**: `QdrantVectorStore` für nahtlose Nutzung
# - **Hybrid Search**: Semantische + Keyword-Suche kombiniert
#   - `RetrievalMode.HYBRID` mit `FastEmbedSparse`
# - **Persistent**: Daten bleiben auf der Festplatte erhalten
# - **Achtung**: Liefert immer Ergebnisse — auch bei irrelevanten Anfragen
#
# **Nächster Schritt**: RAG-System mit LangChain bauen!

# %% [markdown]
#
# ## Workshop-Aufgaben
#
# 1. Installieren und testen Sie Qdrant auf Ihrem Computer
# 2. Erstellen Sie einen Vektor-Store mit `QdrantVectorStore.from_texts()` und
#    mindestens 5 eigenen Dokumenten
# 3. Testen Sie die semantische Suche mit verschiedenen Anfragen
# 4. Probieren Sie eine irrelevante Anfrage aus — was passiert mit den Scores?
# 5. Erstellen Sie einen Hybrid-Vektor-Store mit `FastEmbedSparse`
# 6. Vergleichen Sie semantische Suche vs. Hybrid Search für eine Keyword-Anfrage
