# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: Vektor-Embeddings</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Workshop: Vector Embeddings
#
# In diesem Workshop üben Sie:
# 1. Eigene Texte einbetten und Ähnlichkeiten berechnen
# 2. Eine semantische Suchfunktion bauen
# 3. Keyword-Suche und semantische Suche vergleichen

# %%
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import bm25s
import Stemmer

# %%
load_dotenv()

# %%
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

# %%
bm25_stemmer = Stemmer.Stemmer("german")
bm25_stopwords = "de"

# %% [markdown]
#
# ## Aufgabe 1: Eigene Texte einbetten
#
# Erstellen Sie Embeddings für die folgenden Texte und berechnen Sie die
# Ähnlichkeitsmatrix. Welche Texte sind sich am ähnlichsten?

# %%
workshop_texts = [
    "Hunde sind treue Gefährten und brauchen viel Auslauf",
    "Katzen sind unabhängige Haustiere, die gerne schlafen",
    "Python ist eine beliebte Programmiersprache für Data Science",
    "JavaScript wird hauptsächlich für Webentwicklung eingesetzt",
    "Fußball ist die beliebteste Sportart der Welt",
    "Basketball wurde 1891 in den USA erfunden",
]

# %%

# %%

# %%

# %% [markdown]
#
# ## Aufgabe 2: Semantische Suchfunktion
#
# 1. Schreiben Sie eine Funktion `semantic_search(query, texts, text_embeddings,
#    embeddings_model, k=2)`, die:
#    - Die Anfrage mit `embed_query()` embeddet
#    - Kosinus-Ähnlichkeit zu allen Text-Embeddings berechnet
#    - Die Top-k ähnlichsten Texte mit ihren Scores zurückgibt
# 2. Testen Sie Ihre Funktion mit mindestens 3 verschiedenen Anfragen
#
# **Hinweis:** Sie können dabei die bereits berechneten Embeddings aus Aufgabe 1
# verwenden, um API-Aufrufe zu sparen!

# %%

# %%

# %%
test_queries = [
    "Welches Tier ist ein guter Begleiter?",
    "Programmierung und Software",
    "Sport und Bewegung",
]


# %%

# %% [markdown]
#
# ## Aufgabe 3: Keyword vs. Semantische Suche
#
# Vergleichen Sie die Ergebnisse der Keyword-Suche und der semantischen Suche
# für dieselben Anfragen. Wo liegen die Unterschiede?

# %%
def keyword_search(query, texts):
    return [t for t in texts if query.lower() in t.lower()]

# %%

# %%
comparison_queries = [
    "Haustiere",
    "pets",
    "Programmierung",
    "Ballsport",
]

# %%
