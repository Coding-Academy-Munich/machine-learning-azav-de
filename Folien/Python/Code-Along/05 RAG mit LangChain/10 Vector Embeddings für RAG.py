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
# - In Lektion 04.7 haben wir Embeddings kennengelernt
# - **Embeddings**: Wörter als Vektoren (Listen von Zahlen)
# - Ähnliche Wörter → ähnliche Vektoren
#
# **Für RAG**: Wir brauchen Embeddings für ganze Texte!

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

# %% [markdown]
#
# ## Text-Embeddings
#
# - Nicht nur Wörter, sondern **ganze Texte** einbetten
# - Sätze, Absätze, Dokumente → Vektoren
# - **Dimension**: Meist 384, 768, oder 1536
# - **Modelle**:
#   - OpenAI: text-embedding-ada-002 (1536 Dimensionen)
#   - HuggingFace: verschiedene Modelle

# %% [markdown]
#
# ## Embeddings mit LangChain erstellen

# %%
# Create embedding model
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

# %%
# Example texts
texts = [
    "Machine Learning ist ein Teilbereich der KI",
    "Deep Learning nutzt neuronale Netze",
    "Python ist eine Programmiersprache",
    "Katzen sind Haustiere"
]

# %%

# %%

# %% [markdown]
#
# ## Semantische Ähnlichkeit
#
# - Embeddings ermöglichen **semantische Suche**
# - Nicht nur gleiche Wörter, sondern **ähnliche Bedeutung**
# - **Cosinus-Ähnlichkeit**: Misst Ähnlichkeit zwischen Vektoren
# - Wert von 0 (völlig unterschiedlich) bis 1 (identisch)

# %%
# Calculate similarities
similarities = cosine_similarity(text_embeddings)

# %%

# %% [markdown]
#
# ## Was sehen wir?
#
# - Text 1 und 2 (ML und Deep Learning): **hohe Ähnlichkeit** (~0.8)
# - Text 3 (Python) und 1-2: **mittlere Ähnlichkeit** (~0.5)
# - Text 4 (Katzen): **niedrige Ähnlichkeit** zu allen (~0.2-0.3)
#
# **Das ist semantische Suche!**

# %% [markdown]
#
# ## Semantische Suche in Aktion

# %%
# Query
query = "Was ist neuronale Netze?"

# %%

# %%

# %%

# %% [markdown]
#
# ## Für RAG bedeutet das:
#
# 1. Alle Dokumente embedden
# 2. In Vektor-Datenbank speichern
# 3. Bei Anfrage: Anfrage embedden
# 4. Ähnlichste Dokumente finden
# 5. Als Kontext an LLM geben
#
# **ChromaDB** macht Schritte 2-4 automatisch!

# %% [markdown]
#
# ## Workshop-Aufgaben
#
# 1. Eigene Texte embedden
# 2. Ähnlichkeiten berechnen
# 3. Semantische Suche implementieren
# 4. Mit keyword-Suche vergleichen
# 5. Verschiedene Embedding-Modelle testen

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Embeddings**: Text → Vektoren (Listen von Zahlen)
# - **Semantische Ähnlichkeit**: Via Cosinus-Ähnlichkeit
# - **LangChain**: Einfache Embedding-Erstellung
# - **Semantische Suche**: Bedeutung statt exakte Wörter
# - **Für RAG**: Relevante Dokumente finden
#
# **Nächster Schritt**: ChromaDB für effiziente Vektor-Speicherung!

# %%
