# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Hybride Suche: Semantische + Keyword-Suche</b>
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
from langchain_qdrant import QdrantVectorStore
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
results = semantic_store.similarity_search_with_relevance_scores(
    "Can I integrate individual changes from other versions of the software?", k=2
)

# %%
for doc, score in results:
    print(f"[{score:.2f}] {doc.page_content}")

# %% [markdown]
#
# ## Test 2: Semantische Suche versagt bei Commit-Hashes

# %%
results_hash = semantic_store.similarity_search_with_relevance_scores("commit a3f7bc2", k=2)

# %%
for doc, score in results_hash:
    print(f"[{score:.2f}] {doc.page_content[:80]}...")


# %% [markdown]
#
# ## Die Lösung: Hybride Suche
#
# - **Hybride Suche** kombiniert semantische Suche mit Keyword-Suche (z.B. BM25)
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
#   - Dokumente, die in **einer oder beiden** Listen weit oben stehen, bekommen
#     einen hohen kombinierten Score
#   - Ein Dokument muss nicht in beiden Listen erscheinen
#   - Aber Dokumente, die in beiden Listen auftauchen, bekommen einen Bonus
#
# | Semantische Liste  | Keyword-Liste  | Kombinierte Liste         |
# |--------------------|----------------|---------------------------|
# | 1. Dokument A      | 1. Dokument C  | 1. Dokument A (beide)     |
# | 2. Dokument B      | 2. Dokument A  | 2. Dokument C (beide)     |
# | 3. Dokument C      | 3. Dokument D  | 3. Dokument B (semantisch)|
# |                    |                | 4. Dokument D (keyword)   |

# %% [markdown]
#
# ## Reciprocal Rank Fusion im Detail
#
# - Die Formel für die Berechnung des kombinierten Scores eines Dokuments ist:
#
# $$\text{RRF}(d) = \sum_{s \in S} \frac{1}{k + r_s(d)}$$
#
# - $S$ ist die Menge der Suchverfahren (z.B. semantisch und keyword)
# - $r_s(d)$ ist die Platzierung von Dokument $d$ in der Rangliste von
#   Suchverfahren $s$
# - $k$ ist eine Konstante (typischerweise 60)

# %%
def reciprocal_rank_fusion(*lists: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked_list in lists:
        for rank, doc in enumerate(ranked_list, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)

# %%
reciprocal_rank_fusion(["A", "B", "C"], ["C", "A", "D"])

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Semantische Suche** versteht Bedeutung, kann aber exakte Fachbegriffe
#   verfehlen
# - **Keyword-Suche (BM25)** findet exakte Begriffe, versteht aber keine
#   Bedeutung
# - **Reciprocal Rank Fusion (RRF)** kombiniert die Ranglisten beider Verfahren
# - Hybride Suche nutzt die Stärken beider Ansätze und gleicht die Schwächen aus
#
# **Nächster Schritt**: Hybride Suche mit Qdrant implementieren!
