# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Vector Embeddings für RAG</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Rückblick: Embeddings
#
# - In Sektion 03 (Neuronale Netze) haben wir Embeddings kennengelernt
# - **Embeddings**: Wörter als Vektoren (Listen von Zahlen)
# - Ähnliche Wörter → ähnliche Vektoren
#
# **Für RAG**: Wir brauchen Embeddings für ganze Texte!

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from langchain_openai import OpenAIEmbeddings

# %% [markdown]
#
# ## Text-Embeddings
#
# - Nicht nur Wörter, sondern **ganze Texte** einbetten
# - Sätze, Absätze, Dokumente → Vektoren
# - **Dimension**: Meist 384, 768, oder 1536
#   - Mehr Dimensionen = mehr Nuancen, aber auch teurer
# - **Modelle**:
#   - OpenAI: text-embedding-3-small (1536 Dimensionen)
#   - HuggingFace: verschiedene Open-Source-Modelle
#   - ChromaDB: eingebautes Modell (all-MiniLM-L6-v2)

# %% [markdown]
#
# ## Embeddings mit LangChain erstellen

# %%
# TODO: Create an OpenAIEmbeddings model using OpenRouter
# embeddings = OpenAIEmbeddings(...)

# %%
texts = [
    "Machine Learning ist ein Teilbereich der KI",
    "Deep Learning nutzt neuronale Netze",
    "Python ist eine Programmiersprache",
    "Katzen sind Haustiere"
]

# %%
# TODO: Create embeddings for all texts
# text_embeddings = ...

# %%

# %% [markdown]
#
# ## Semantische Ähnlichkeit
#
# - Embeddings ermöglichen **semantische Suche**
# - Nicht nur gleiche Wörter, sondern **ähnliche Bedeutung**
# - Aber wie messen wir die Ähnlichkeit zwischen zwei Vektoren?
# - Zwei Ansätze:
#   - **Euklidische Distanz**: Abstand zwischen zwei Punkten
#   - **Kosinus-Ähnlichkeit**: Winkel zwischen zwei Vektoren

# %% [markdown]
#
# ## Euklidische Distanz vs. Kosinus-Ähnlichkeit
#
# <div style="float:right;width:40%;">
#   <img src="img/cosine-distance.png" style="float:right;width:50%"/>
#   <img src="img/vector-difference.png" style="float:left;width:50%"/>
# </div>
# <div style="width:60%;">
# <br>
# <ul>
#   <li><b>Euklidische Distanz</b>: Wie weit sind zwei Punkte voneinander entfernt?</li>
#   <ul>
#     <li>Misst den "Abstand" ($|\vec a - \vec b|$)</li>
#     <li>Problematisch: Längere Texte haben andere Vektoren als kurze</li>
#   </ul>
#   <li><b>Kosinus-Ähnlichkeit</b>: Zeigen zwei Pfeile in die gleiche Richtung?</li>
#   <ul>
#     <li>Misst den Winkel zwischen Vektoren</li>
#     <li>Unabhängig von der "Länge" des Vektors</li>
#     <li>Für Text <b>besser geeignet</b>!</li>
#   </ul>
# </ul>
# </div>

# %% [markdown]
#
# ## Warum Kosinus-Ähnlichkeit für Text?
#
# - Zwei Texte mit gleicher Bedeutung aber unterschiedlicher Länge:
#   - Kurz: "ML lernt aus Daten"
#   - Lang: "Machine Learning ist ein Verfahren, das automatisch aus Daten lernt"
# - **Euklidische Distanz**: Groß (verschiedene Vektorlängen)
# - **Kosinus-Ähnlichkeit**: Hoch (gleiche Richtung = gleiche Bedeutung!)
#
# **Wert**: 0.0 (völlig unterschiedlich) bis 1.0 (identisch)

# %% [markdown]
#
# ## Ähnlichkeitsmatrix berechnen

# %%
similarities = cosine_similarity(text_embeddings)

# %%

# %% [markdown]
#
# ## Was sehen wir?
#
# - Text 1 und 2 (ML und Deep Learning): **hohe Ähnlichkeit** (~0.8)
#   - Beide handeln von KI-Methoden
# - Text 3 (Python) und 1–2: **mittlere Ähnlichkeit** (~0.5)
#   - Python wird oft für ML genutzt — das "weiß" das Modell!
# - Text 4 (Katzen): **niedrige Ähnlichkeit** zu allen (~0.2–0.3)
#   - Hat nichts mit Technologie zu tun
#
# **Das ist die Stärke der semantischen Suche!**

# %% [markdown]
#
# ## Semantische Suche in Aktion

# %%
query = "Was ist neuronale Netze?"


# %%
# TODO: Embed the query and find the most similar texts
# 1. Embed the query with embeddings.embed_query(query)
# 2. Calculate cosine similarity
# 3. Sort by similarity

# %%

# %% [markdown]
#
# ## Warum nicht einfach Keyword-Suche?
#
# - **Keyword-Suche**: Sucht nach exakten Wörtern im Text
# - **Problem**: Findet nur exakte Übereinstimmungen
# - Versteht keine Synonyme, keine Übersetzungen, keine verwandten Konzepte

# %%
def keyword_search(query, texts):
    """Simple keyword search: returns texts containing the query."""
    return [t for t in texts if query.lower() in t.lower()]

# %% [markdown]
#
# ## Vergleich: Keyword vs. Semantische Suche

# %%

# %%

# %% [markdown]
#
# ## Keyword-Suche versagt bei anderen Sprachen

# %%

# %%

# %% [markdown]
#
# ## Semantische Suche versteht Bedeutung
#
# | | Keyword-Suche | Semantische Suche |
# |---|---|---|
# | **Exakte Wörter** | ✅ Findet sie | ✅ Findet sie |
# | **Synonyme** | ❌ "KI" ≠ "AI" | ✅ Versteht Bedeutung |
# | **Andere Sprache** | ❌ "AI" ≠ "KI" | ✅ Sprachübergreifend |
# | **Verwandte Konzepte** | ❌ Kein Zusammenhang | ✅ Erkennt Verwandtschaft |

# %% [markdown]
#
# ## Für RAG bedeutet das:
#
# 1. Alle Dokument-Chunks embedden
# 2. In Vektor-Datenbank speichern
# 3. Bei Anfrage: Anfrage embedden
# 4. Ähnlichste Chunks finden (Kosinus-Ähnlichkeit)
# 5. Als Kontext an LLM geben
#
# **ChromaDB** macht Schritte 2–4 automatisch!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Text-Embeddings**: Ganze Texte als Vektoren darstellen
# - **Kosinus-Ähnlichkeit**: Misst "Richtung" (Bedeutung), nicht "Länge"
#   - Besser als euklidische Distanz für Text
#   - Wert: 0.0 (unterschiedlich) bis 1.0 (identisch)
# - **Semantische Suche**: Versteht Bedeutung, nicht nur exakte Wörter
#   - Findet Synonyme, verwandte Konzepte, andere Sprachen
# - **LangChain**: Einfache Embedding-Erstellung mit wenigen Zeilen
#
# **Nächster Schritt**: ChromaDB für effiziente Vektor-Speicherung!
