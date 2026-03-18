# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: Vektor-Datenbanken</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Workshop: Vektor-Datenbanken
#
# In diesem Workshop üben Sie:
# 1. Einen Vektor-Store mit eigenen Dokumenten erstellen
# 2. Distanz-Scores interpretieren und irrelevante Anfragen filtern
# 3. Einen Hybrid-Vektor-Store erstellen und vergleichen

# %%
# !pip install --root-user-action=ignore --quiet qdrant-client langchain-qdrant

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

# %%

# %% [markdown]
#
# ## Aufgabe 2: Distanz-Scores und Filterung
#
# Testen Sie eine irrelevante Anfrage und vergleichen Sie die Distanz-Scores
# mit einer relevanten Anfrage. Implementieren Sie eine Filterung nach
# Distanz-Schwellenwert.

# %%

# %%
threshold = 0.5

# %%

# %% [markdown]
#
# ## Aufgabe 3: Hybride Suche
#
# Erstellen Sie einen Hybrid-Vektor-Store mit denselben Dokumenten.
# Vergleichen Sie die Ergebnisse der semantischen Suche mit der hybriden Suche
# für eine Anfrage, die einen spezifischen Begriff enthält.

# %%
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# %%

# %%
query = "K2"

# %%
