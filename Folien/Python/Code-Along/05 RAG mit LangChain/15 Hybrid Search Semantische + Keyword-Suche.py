# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Hybrid Search: Semantische + Keyword-Suche</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Rückblick: Semantische Suche
#
# - In der letzten Lektion haben wir Qdrant für **semantische Suche** genutzt
# - Semantische Suche versteht **Bedeutung**: Synonyme, verwandte Konzepte, andere Sprachen
# - Aber: Hat semantische Suche auch **Schwächen**?

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
# ## Wann versagt semantische Suche?
#
# - **Spezifische Fachbegriffe**: Seltene Abkürzungen oder Akronyme, die das
#   Embedding-Modell nicht gut gelernt hat
# - **Identifikatoren**: Fehlercodes, Modellnamen, Versionsnummern, Produktnummern
# - **Exakte Übereinstimmung nötig**: Wenn der Nutzer genau dieses Wort meint,
#   nicht etwas Ähnliches
#
# Das Embedding-Modell bildet seltene Begriffe manchmal auf verwandte aber
# **falsche** Konzepte ab

# %% [markdown]
#
# ## Demonstration: Semantische Suche versagt
#
# - Wir erstellen Dokumente über ML-Konzepte
# - Einige enthalten spezifische Fachbegriffe (MSE, RMSE, MAE)
# - Wir testen, ob semantische Suche die richtigen Dokumente findet

# %%
ml_docs = [
    "Linear regression fits a straight line through data points using the normal equation.",
    "The MSE (Mean Squared Error) measures the average of squared prediction errors. RMSE is the square root of MSE.",
    "Neural networks learn by adjusting weights through backpropagation and gradient descent.",
    "Cross-validation splits data into training and test sets to evaluate model performance.",
    "The MAE (Mean Absolute Error) measures average absolute prediction errors. Unlike MSE, it is less sensitive to outliers.",
    "Regularization techniques like L1 and L2 prevent overfitting by adding penalty terms.",
    "Decision trees split data based on feature values to make predictions.",
]

# %%
semantic_store = QdrantVectorStore.from_texts(
    texts=ml_docs,
    embedding=embeddings,
    collection_name="semantic_demo",
    location=":memory:",
)

# %% [markdown]
#
# ## Test 1: Semantische Suche funktioniert gut

# %%

# %% [markdown]
#
# ## Test 2: Semantische Suche versagt bei Fachbegriffen

# %%

# %% [markdown]
#
# - Die semantische Suche findet möglicherweise nicht das richtige Dokument
# - "RMSE" wird als allgemeines Konzept abgebildet, nicht als exakter Begriff
# - Das Dokument, das RMSE buchstäblich enthält, wird möglicherweise nicht
#   an erster Stelle zurückgegeben

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
# ! pip install fastembed-gpu

# %%
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# %% [markdown]
#
# ## Hybrid-Vektor-Store erstellen

# %%
# TODO: Create QdrantVectorStore with RetrievalMode.HYBRID
# using both embeddings and sparse_embeddings

# %%

# %% [markdown]
#
# ## Vergleich: Semantische Suche vs. Hybrid Search
#
# Testen wir mit der Anfrage, bei der semantische Suche Probleme hatte:

# %%
query = "RMSE"

# %% [markdown]
#
# Semantische Suche:

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
# - Hybrid Search hat einen geringen zusätzlichen Rechenaufwand
#   (Sparse-Embedding-Berechnung), aber die Verbesserung der Abrufqualität
#   ist erheblich

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
