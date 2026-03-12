# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Hybrid Search: Semantische + Keyword-Suche</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Rückblick: Semantische Suche
#
# - Qdrant für **semantische Suche**
# - Semantische Suche versteht Bedeutung: Synonyme, verwandte Konzepte,
#   andere Sprachen
# - Aber: Hat semantische Suche auch Schwächen?

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

# %% [markdown]
#
# ## Wann versagt semantische Suche?
#
# - **Spezifische Fachbegriffe**: Seltene Abkürzungen oder Akronyme, die das
#   Embedding-Modell nicht gut gelernt hat
# - **Identifikatoren**:
#   - Fehlercodes (`CUDA_ERROR_OUT_OF_MEMORY`)
#   - Modellnamen (`ministral-14b`)
#   - Versions-, Produkt- oder Buildnummern (`v1.2.3`, `build-4567`)
# - **Exakte Übereinstimmung nötig**: Wenn der Nutzer genau dieses Wort meint,
#   nicht etwas Ähnliches
#
# → Embedding-Modell bildet seltene Begriffe manchmal auf **falsche** Konzepte ab

# %% [markdown]
#
# ## Demonstration: Semantische Suche versagt
#
# - Wir erstellen Dokumente über Git und Commits
# - Eines enthält einen spezifischen Commit-Hash (`a3f7bc2`)
# - Wir testen, ob semantische Suche das richtige Dokument findet

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
# ## Test 1: Semantische Suche funktioniert gut

# %%
results = semantic_store.similarity_search(
    "How do I undo changes in git?", k=2
)

# %%

# %% [markdown]
#
# ## Test 2: Semantische Suche versagt bei Commit-Hashes

# %%
results_hash = semantic_store.similarity_search("commit a3f7bc2", k=2)

# %% [markdown]
# Abfrage: "commit a3f7bc2"

# %%

# %% [markdown]
#
# - Die semantische Suche findet nicht das richtige Dokument
# - "commit" wird verstanden, aber "a3f7bc2" ist für das Embedding-Modell
#   eine bedeutungslose Zeichenkette
# - Das Dokument mit dem exakten Hash wird nicht an erster Stelle
#   zurückgegeben

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
# ## Die Lösung: Hybrid Search
#
# - **Hybrid Search** kombiniert semantische Suche mit Keyword-Suche (BM25)
# - Zwei unabhängige Suchverfahren laufen **parallel**:
#   - **Dense Embeddings** (semantisch): Versteht Bedeutung
#   - **Sparse Embeddings** (BM25): Findet exakte Wörter
# - Die Ergebnisse werden zu einer gemeinsamen Rangliste kombiniert
#
# Ziel: Die Stärken beider Verfahren nutzen, die Schwächen ausgleichen

# %% [markdown]
#
# ## Wie werden die Ergebnisse kombiniert?
#
# - Qdrant verwendet **Reciprocal Rank Fusion (RRF)**
# - Prinzip:
#   - Jedes Suchverfahren liefert eine Rangliste
#   - Dokumente, die in **einer oder beiden** Listen weit oben stehen,
#     bekommen einen hohen kombinierten Score
#   - Ein Dokument muss nicht in beiden Listen erscheinen
#
# ```
# Semantische Liste:     Keyword-Liste:       Kombinierte Liste:
# 1. Dokument A          1. Dokument C        1. Dokument C (beide)
# 2. Dokument B          2. Dokument A        2. Dokument A (beide)
# 3. Dokument C          3. Dokument D        3. Dokument B (semantisch)
#                                              4. Dokument D (keyword)
# ```

# %% [markdown]
#
# ## Hybrid Search mit Qdrant
#
# - Qdrant unterstützt Hybrid Search nativ
# - Braucht zwei Arten von Embeddings:
#   - **Dense Embeddings**: Für semantische Suche (wie bisher)
#   - **Sparse Embeddings**: Für Keyword-Suche (BM25)
# - `FastEmbedSparse` erzeugt die Sparse Embeddings

# %%
# ! pip install fastembed

# %%
sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")

# %% [markdown]
#
# ## Hybrid-Vektor-Store erstellen

# %% [markdown]
#
# Erstellen Sie einen `QdrantVectorStore` mit `RetrievalMode.HYBRID`
# unter Verwendung von `embedding_model` und `sparse_embedding`:

# %%

# %% [markdown]
#
# ## Vergleich: Semantische Suche vs. Hybrid Search
#
# Testen wir mit der Anfrage, bei der semantische Suche Probleme hatte:

# %%
query = "commit a3f7bc2"

# %% [markdown]
#
# Semantische Suche:

# %%

# %%

# %% [markdown]
#
# Hybrid Search:

# %%

# %% [markdown]
#
# Semantische Anfrage funktioniert weiterhin:

# %%

# %% [markdown]
#
# ## Wann welches Suchverfahren verwenden?
#
# - **Rein semantisch**: Akzeptabel für schnelle Prototypen oder wenn der
#   Dokumentenkorpus klein und homogen ist
# - **Hybrid Search**: Empfohlen für jedes Produktivsystem, besonders wenn
#   Dokumente Fachbegriffe, Identifikatoren oder domänenspezifische
#   Abkürzungen enthalten
# - Geringer Mehraufwand, deutliche Qualitätsverbesserung

# %% [markdown]
#
# ## Hybrid Search in der Praxis
#
# - Hybrid Search ist in Produktiv-RAG-Systemen **Standard-Praxis**
# - Die meisten RAG-Fehler in der Praxis entstehen durch
#   schlechtes Retrieval — die falsche Information wird abgerufen
# - Hybrid Search reduziert diese Fehler deutlich
# - Daher verwenden wir ab jetzt immer `RetrievalMode.HYBRID`

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Semantische Suche** versteht Bedeutung, kann aber exakte Fachbegriffe verfehlen
# - **Keyword-Suche (BM25)** findet exakte Begriffe, versteht aber keine Bedeutung
# - **Hybrid Search** kombiniert beide Verfahren über Reciprocal Rank Fusion
# - Qdrant unterstützt Hybrid Search nativ mit `FastEmbedSparse` und `RetrievalMode.HYBRID`
# - Hybrid Search ist Standard-Praxis für Produktiv-RAG-Systeme
#
# **Nächster Schritt**: RAG-System mit LangChain bauen!
