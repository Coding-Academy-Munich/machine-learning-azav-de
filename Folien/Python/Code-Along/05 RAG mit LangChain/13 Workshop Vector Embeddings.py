# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: Vector Embeddings</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

# %%
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

# %% [markdown]
#
# ## Aufgabe 1: Eigene Texte einbetten
#
# Erstellen Sie Embeddings für die folgenden Texte und berechnen Sie die
# Ähnlichkeitsmatrix. Welche Texte sind sich am ähnlichsten?

# %%
workshop_texts = [
    "Hunde sind treue Haustiere und brauchen viel Auslauf",
    "Katzen sind unabhängige Haustiere, die gerne schlafen",
    "Python ist eine beliebte Programmiersprache für Data Science",
    "JavaScript wird hauptsächlich für Webentwicklung eingesetzt",
    "Fußball ist die beliebteste Sportart der Welt",
    "Basketball wurde 1891 in den USA erfunden",
]


# %%
# Erstellen Sie Embeddings und berechnen Sie die Ähnlichkeitsmatrix
# Create embeddings and compute the similarity matrix

# TODO: Embed all texts
# TODO: Calculate cosine similarities
# TODO: Print the most similar pair of texts

# %%

# %% [markdown]
#
# ## Aufgabe 2: Semantische Suchfunktion
#
# Schreiben Sie eine Funktion `semantic_search(query, texts, embeddings_model, k=2)`,
# die:
# 1. Die Anfrage embeddet
# 2. Kosinus-Ähnlichkeit zu allen Texten berechnet
# 3. Die Top-k ähnlichsten Texte mit ihren Scores zurückgibt
#
# Testen Sie Ihre Funktion mit mindestens 3 verschiedenen Anfragen.

# %%
def semantic_search(query, texts, embeddings_model, k=2):
    """Semantische Suche / Semantic search."""
    # TODO: Embed the query
    # TODO: Embed all texts (or reuse existing embeddings)
    # TODO: Calculate cosine similarity
    # TODO: Return top-k results as list of (score, text) tuples
    pass

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
    """Simple keyword search: returns texts containing the query."""
    return [t for t in texts if query.lower() in t.lower()]

# %%
# Vergleichen Sie beide Suchansätze mit den folgenden Anfragen:
# Compare both search approaches with the following queries:
comparison_queries = [
    "Haustiere",
    "pets",
    "Programmierung",
    "Ballsport",
]

# TODO: Für jede Anfrage: Keyword-Suche und semantische Suche ausführen
# TODO: Ergebnisse vergleichen und ausgeben
# TODO: For each query: Run keyword search and semantic search
# TODO: Compare and print results
