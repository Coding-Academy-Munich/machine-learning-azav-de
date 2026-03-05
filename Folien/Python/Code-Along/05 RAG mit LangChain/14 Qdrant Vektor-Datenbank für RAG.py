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
# - **Problem**: Semantische Suche über viele Dokumente ist aufwändig
#   - Bei jedem Query: alle Dokumente embedden + Ähnlichkeit berechnen
# - **Lösung**: Vektor-Datenbank
#   - Speichert Embeddings einmal vor
#   - Nutzt spezielle Indexstrukturen (z.B. HNSW) für schnelle Suche
#   - Effizienz hängt von Konfiguration und Hardware ab, aber auch große
#     Sammlungen können in wenigen Millisekunden durchsucht werden

# %% [markdown]
#
# ## Qdrant
#
# - **Open Source** (Apache 2.0 Lizenz)
# - Einfach zu installieren: `pip install qdrant-client langchain-qdrant`
# - Funktioniert auf Windows und Linux
# - Lokaler Modus ohne Server-Konfiguration
# - Unterstützt Hybrid Search (semantische + Keyword-Suche)
# - LangChain-Integration verfügbar

# %% [markdown]
#
# ## Installation
#
# ```bash
# pip install qdrant-client langchain-qdrant
# ```
#
# Keine Docker-Installation und keine Server-Konfiguration nötig.

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
# 1. **In-Memory**: Daten nur im RAM (für Tests und Experimente)
#    - `location=":memory:"`
# 2. **Persistent**: Daten auf Festplatte (für echte Anwendungen)
#    - `path="./qdrant_db"`
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
# ## Suche mit Ähnlichkeits-Scores
#
# - `similarity_search_with_score()` zeigt auch einen Score
# - Der Score ist die **Kosinus-Ähnlichkeit**
# - **Hoher Score** = **hohe Ähnlichkeit**
#
# | Score | Bedeutung |
# |---|---|
# | 0.90+ | Sehr ähnlich |
# | 0.50 | Mittelmäßig ähnlich |
# | 0.10 | Wenig ähnlich |

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
# - **Niedriger Score** = niedrige Relevanz
# - Wir können nach Ähnlichkeit filtern, um irrelevante Ergebnisse auszuschließen

# %% [markdown]
#
# ## Ergebnisse nach Ähnlichkeit filtern
#
# - Einen Schwellenwert setzen: Nur Ergebnisse mit hoher Ähnlichkeit behalten
# - Schwellenwert hängt von der Anwendung ab (typisch: 0.3–0.5)

# %%
threshold = 0.4

# %%

# %%

# %% [markdown]
#
# ## Metadata verwenden
#
# - Dokumente können **Metadata** haben (z.B. Thema, Datum, Quelle)
# - Nützlich für zusätzliche Filterung:
#   - Nur Dokumente eines bestimmten Themas durchsuchen
#   - Nach Datum filtern
#   - Quelle in der Antwort anzeigen

# %%
from langchain_core.documents import Document

docs_with_meta = [
    Document(page_content="Linear regression models linear relationships",
             metadata={"topic": "ML", "difficulty": "beginner"}),
    Document(page_content="Python is a popular programming language",
             metadata={"topic": "Programming", "difficulty": "beginner"}),
    Document(page_content="Decision trees split data into branches",
             metadata={"topic": "ML", "difficulty": "beginner"}),
    Document(page_content="Gradient boosting combines many weak learners",
             metadata={"topic": "ML", "difficulty": "advanced"}),
]

vectorstore_meta = QdrantVectorStore.from_documents(
    documents=docs_with_meta,
    embedding=embeddings,
    collection_name="docs_with_meta",
    location=":memory:",
)

# %% [markdown]
#
# ## Suche mit Metadata

# %%

# %% [markdown]
#
# ## Metadata-Filterung
#
# - Mit `filter` kann die Suche auf bestimmte Metadata-Werte eingeschränkt werden
# - Qdrant verwendet dafür eigene Filter-Objekte

# %%
from qdrant_client.models import Filter, FieldCondition, MatchValue

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
# ## Zusammenfassung
#
# - **Qdrant**: Open-Source Vektor-Datenbank
# - Einfach installierbar, keine Server-Konfiguration nötig
# - **LangChain-Integration**: `QdrantVectorStore` für nahtlose Nutzung
# - **Ähnlichkeits-Scores**: Hoch = ähnlich (Kosinus-Ähnlichkeit)
# - **Metadata**: Dokumente können Metadaten haben und danach gefiltert werden
# - **Persistent**: Daten bleiben auf der Festplatte erhalten
# - **Achtung**: Liefert immer Ergebnisse — Ähnlichkeits-Filter verwenden!
#
# **Nächster Schritt**: Hybrid Search — semantische + Keyword-Suche kombinieren!
