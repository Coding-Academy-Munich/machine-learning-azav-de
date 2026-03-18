# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Hybride Suche mit Qdrant</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %%
# ! pip install --root-user-action=ignore --quiet fastembed

# %%
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import OpenAIEmbeddings

# %%
load_dotenv()

# %%
embedding_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small",
)

# %%
docs = [
    "The bug introduced in commit a3f7bc2 caused the tokenizer to silently drop special tokens during batch encoding.",
    "Git tracks changes through a directed acyclic graph of commits, where each commit stores a snapshot of the repository state along with metadata.",
    "Continuous integration runs automated tests on every commit to catch regressions before they reach the main branch.",
    "Debugging machine learning pipelines often requires bisecting commits to find where model performance regressed.",
    "Cherry-picking individual commits allows developers to selectively apply bug fixes from one branch to another without merging all changes.",
    "The git blame command shows which commit last modified each line of a file, helping developers understand the history behind code changes.",
    "Squashing multiple commits into one creates a cleaner history by combining related changes into a single commit before merging.",
]

# %%
semantic_store = QdrantVectorStore.from_texts(
    texts=docs,
    embedding=embedding_model,
    collection_name="semantic_demo",
    location=":memory:",
)

# %% [markdown]
#
# ## Hybride Suche mit Qdrant
#
# - Qdrant unterstützt hybride Suche nativ
# - Braucht zwei Arten von Embeddings:
#   - **Dense Embeddings**: Für semantische Suche (wie bisher)
#   - **Sparse Embeddings**: Für Keyword-Suche (BM25)
# - `FastEmbedSparse` erzeugt die Sparse Embeddings

# %%
sparse_embedding_model = FastEmbedSparse(model_name="Qdrant/bm25")

# %% [markdown]
#
# ## Hybrid-Vektor-Store erstellen

# %%
hybrid_store = QdrantVectorStore.from_texts(
    texts=docs,
    embedding=embedding_model,
    sparse_embedding=sparse_embedding_model,
    collection_name="hybrid_demo",
    location=":memory:",
    retrieval_mode=RetrievalMode.HYBRID,
)

# %% [markdown]
#
# ## Vergleich: Semantische Suche vs. Hybride Suche
#
# Testen wir mit der Anfrage, bei der semantische Suche Probleme hatte:

# %%
query = "commit a3f7bc2"

# %% [markdown]
#
# Semantische Suche:

# %%
results_semantic = semantic_store.similarity_search_with_relevance_scores(query, k=2)

# %%
for doc, score in results_semantic:
    print(f"[{score:.2f}] {doc.page_content[:80]}...")

# %% [markdown]
#
# Hybride Suche:

# %%
results_hybrid = hybrid_store.similarity_search_with_relevance_scores(query, k=2)

# %%
for doc, score in results_hybrid:
    print(f"[{score:.2f}] {doc.page_content[:80]}...")

# %% [markdown]
#
# Semantische Anfrage funktioniert weiterhin:

# %%
results_semantic_query = hybrid_store.similarity_search_with_relevance_scores(
    "Can I integrate individual changes from other versions of the software?", k=2
)

# %%
for doc, score in results_semantic_query:
    print(f"[{score:.2f}] {doc.page_content[:80]}...")

# %% [markdown]
#
# ## Stärken und Schwächen der Suchverfahren
#
# | | Semantische Suche | Keyword-Suche (BM25) |
# |---|---|---|
# | **Synonyme** | "KI" findet "AI" | Nur exakter Begriff |
# | **Paraphrasen** | "Modell lernt" ≈ "Training" | Kein Zusammenhang |
# | **Andere Sprache** | "dog" findet "Hund" | Kein Zusammenhang |
# | **Fachbegriffe** | "MSE" → evtl. falsche Treffer | "MSE" → exakte Treffer |
# | **Fehlercodes** | "CUDA_ERROR" → ungenau | "CUDA_ERROR" → exakt |
# | **Modellnamen** | "ministral-14b" → ungenau | "ministral-14b" → exakt |

# %% [markdown]
#
# ## Hybride Suche in der Praxis
#
# - Hybride Suche ist in Produktiv-RAG-Systemen Standard-Praxis
# - Viele RAG-Fehler in der Praxis entstehen durch schlechtes Retrieval
# - Hybride Suche reduziert diese Fehler deutlich
# - **Aber:** Andere Verbesserungen und Tuning sind oft notwendig

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Semantische Suche** versteht Bedeutung, kann aber exakte Fachbegriffe
#   verfehlen
# - **Keyword-Suche (BM25)** findet exakte Begriffe, versteht aber keine
#   Bedeutung
# - **Hybride Suche** kombiniert beide Verfahren (z. B. über Reciprocal Rank
#   Fusion)
# - Qdrant unterstützt hybride Suche nativ mit `FastEmbedSparse` und
#   `RetrievalMode.HYBRID`
# - Hybride Suche ist Standard-Praxis für Produktiv-RAG-Systeme
#
# **Nächster Schritt**: RAG-System mit LangChain bauen!
