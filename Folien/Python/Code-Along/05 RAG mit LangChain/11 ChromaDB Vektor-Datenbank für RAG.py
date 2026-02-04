# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>ChromaDB: Vektor-Datenbank für RAG</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Warum Vektor-Datenbanken?
#
# - **Problem**: Viele Dokumente durchsuchen ist langsam
# - **Lösung**: Vektor-Datenbank
# - Speichert Embeddings effizient
# - Findet ähnlichste Vektoren schnell
# - **Millionen** von Dokumenten in Millisekunden!

# %% [markdown]
#
# ## Warum ChromaDB?
#
# - ✅ **Open Source** (Apache 2.0 Lizenz)
# - ✅ **Einfach zu installieren**: `pip install chromadb`
# - ✅ **Funktioniert auf Windows und Linux**
# - ✅ **Keine Server-Konfiguration** nötig
# - ✅ **Perfekt für Lernen** und Prototypen
# - ✅ **LangChain-Integration** out-of-the-box
#
# **Ideal für Einstieg!**

# %% [markdown]
#
# ## Installation
#
# ```bash
# pip install chromadb
# ```
#
# **Das war's!** Keine Docker, keine Server-Konfiguration.

# %%
# ! pip install chromadb

# %%
import os
import chromadb
from chromadb.config import Settings

# %% [markdown]
#
# ## ChromaDB Modi
#
# 1. **In-Memory**: Daten nur im RAM (für Tests)
# 2. **Persistent**: Daten auf Festplatte (empfohlen)
#
# **Für RAG**: Persistent nutzen!

# %% [markdown]
#
# ## Client erstellen

# %%
# In-memory client (for testing)
# client = chromadb.Client()

# Persistent client (recommended)
# TODO: Create PersistentClient

# %%

# %% [markdown]
#
# ## Collections: Dokument-Sammlungen
#
# - **Collection**: Gruppe von Dokumenten
# - Wie eine Tabelle in einer Datenbank
# - Jede Collection hat:
#   - Dokumente (Texte)
#   - Embeddings (automatisch erstellt!)
#   - Metadata (optionale Zusatzinfos)

# %% [markdown]
#
# ## Collection erstellen

# %%
# TODO: Create or get collection

# %%

# %% [markdown]
#
# ## Dokumente hinzufügen
#
# ChromaDB macht Embeddings **automatisch**!

# %%
# Add documents
documents = [
    "Neural networks can learn complex patterns",
    "Gradient descent optimizes model parameters",
    "Overfitting occurs when models memorize training data",
    "RAG combines retrieval with generation"
]

# TODO: Add to collection

# %%

# %% [markdown]
#
# ## Dokumente suchen

# %%
# TODO: Query collection

# %%

# %% [markdown]
#
# ## Metadata filtern
#
# - Zusätzliche Filter für Suche
# - Beispiel: Nur Dokumente zu bestimmtem Thema
# - Oder: Nur neuere Dokumente

# %%
# Query with metadata filter
results_filtered = collection.query(
    query_texts=["machine learning algorithms"],
    n_results=3,
    where={"topic": "ML"}  # Filter by metadata
)

# %%

# %% [markdown]
#
# ## ChromaDB mit LangChain
#
# - LangChain hat **direkte Integration** mit ChromaDB
# - Noch einfacher zu nutzen!
# - Perfekt für RAG-Systeme

# %%
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# %% [markdown]
#
# Erzeugen des Vektor-Stores mit LangChain

# %%

# %% [markdown]
#
# Dokumente zum Vektor-Store hinzufügen

# %%
texts_lc = [
    "LangChain simplifies LLM application development",
    "Vector stores enable semantic search",
    "Chains combine multiple processing steps"
]

# %%

# %%

# %%

# %% [markdown]
#
# ## Workshop-Aufgaben
#
# 1. ChromaDB installieren und einrichten
# 2. Collection erstellen
# 3. Dokumente hinzufügen
# 4. Semantische Suche testen
# 5. Mit LangChain-Integration arbeiten
# 6. Verschiedene Queries ausprobieren

# %% [markdown]
#
# ## Zusammenfassung
#
# - **ChromaDB**: Open-Source Vektor-Datenbank
# - **Einfach**: Pip-Install, keine Konfiguration
# - **Funktioniert**: Windows, Linux, Mac
# - **Automatisch**: Erstellt Embeddings selbst
# - **LangChain-Integration**: Perfekt für RAG
# - **Persistent**: Daten bleiben erhalten
#
# **Nächster Schritt**: RAG-System mit LangChain bauen!

# %%
