# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Qdrant: Metadaten und Retriever</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Setup

# %%
# !pip install --root-user-action=ignore --quiet qdrant-client langchain-qdrant

# %%
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
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
# ## Metadaten verwenden
#
# - Dokumente können **Metadaten** haben (z.B. Thema, Datum, Quelle)
# - Nützlich für zusätzliche Filterung:
#   - Nur Dokumente eines bestimmten Themas durchsuchen
#   - Nach Datum filtern
#   - Quelle in der Antwort anzeigen

# %%
docs_with_meta = [
    Document(
        page_content="Linear regression models linear relationships",
        metadata={"topic": "ML", "difficulty": "beginner"},
    ),
    Document(
        page_content="Python is a popular programming language",
        metadata={"topic": "Programming", "difficulty": "beginner"},
    ),
    Document(
        page_content="Decision trees split data into branches",
        metadata={"topic": "ML", "difficulty": "beginner"},
    ),
    Document(
        page_content="Gradient boosting combines many weak learners",
        metadata={"topic": "ML", "difficulty": "advanced"},
    ),
]

# %%
vectorstore = QdrantVectorStore.from_documents(
    documents=docs_with_meta,
    embedding=embedding_model,
    collection_name="docs_with_meta",
    location=":memory:",
)

# %% [markdown]
#
# ## Suche mit Metadata

# %%

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

# %%

# %%

# %% [markdown]
#
# ## Retriever erstellen
#
# - Ein **Retriever** ist die Standard-Schnittstelle in LangChain
# - Wird für RAG-Chains verwendet
# - Einfach aus dem Vektor-Store erstellen

# %%

# %%

# %% [markdown]
#
# Gefundene Dokumente:

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
# - **Metadata**: Dokumente können strukturierte Informationen haben (Thema, Datum, Quelle)
# - **Metadata-Filterung**: Suchraum mit Qdrant-Filtern einschränken
# - **Retriever**: Standard-LangChain-Schnittstelle für RAG-Chains
# - **Persistenter Speicher**: `path=` statt `location=":memory:"` für Produktion
#
# **Nächster Schritt**: Hybrid Search — semantische + Keyword-Suche kombinieren!
