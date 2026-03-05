# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: Vektor-Datenbanken</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Workshop: Vektor-Datenbanken
#
# In diesem Workshop üben Sie:
# 1. Einen Vektor-Store mit eigenen Dokumenten erstellen
# 2. Semantische Suche testen und Distanz-Scores interpretieren
# 3. Irrelevante Anfragen erkennen und filtern
# 4. Einen Hybrid-Vektor-Store erstellen und vergleichen

# %%
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import OpenAIEmbeddings

# %%
load_dotenv()

# %%
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

# %% [markdown]
#
# ## Aufgabe 1: Eigene Dokumente
#
# Erstellen Sie einen Vektor-Store mit mindestens 5 Dokumenten zu einem Thema
# Ihrer Wahl (z.B. Kochen, Sport, Geschichte, Musik...).
# Testen Sie die semantische Suche mit 2-3 verschiedenen Anfragen.

# %%
my_texts = [
    # TODO: Add at least 5 texts about a topic
]

# TODO: Create a QdrantVectorStore from your texts
# my_store = QdrantVectorStore.from_texts(...)

# TODO: Test semantic search with 2-3 queries
# results = my_store.similarity_search(...)

# %%

# %% [markdown]
#
# ## Aufgabe 2: Distanz-Scores und Filterung
#
# Testen Sie eine irrelevante Anfrage und vergleichen Sie die Distanz-Scores
# mit einer relevanten Anfrage. Implementieren Sie eine Filterung nach
# Distanz-Schwellenwert.

# %%
# TODO: Search with a relevant query using similarity_search_with_score()
# TODO: Search with an irrelevant query (e.g., about cooking or programming)
# TODO: Compare the scores
# TODO: Implement a threshold filter to discard irrelevant results

# %%

# %% [markdown]
#
# ## Aufgabe 3: Hybrid Search
#
# Erstellen Sie einen Hybrid-Vektor-Store mit denselben Dokumenten.
# Vergleichen Sie die Ergebnisse der semantischen Suche mit der Hybrid Search
# für eine Anfrage, die einen spezifischen Begriff enthält.

# %%
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# TODO: Create a hybrid vector store using RetrievalMode.HYBRID
# my_hybrid_store = QdrantVectorStore.from_texts(...)

# TODO: Compare semantic search vs. hybrid search for a specific term
# e.g., "K2" or "Kilimandscharo"

# %%
