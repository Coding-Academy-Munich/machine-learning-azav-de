# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Qdrant: Vektor-Datenbank für RAG</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Warum Vektor-Datenbanken?
#
# - **Problem**: Semantische Suche über viele Dokumente ist aufwändig
#   - Bei jedem Query: Ähnlichkeit für alle Dokumente berechnen
# - **Lösung**: Vektor-Datenbank
#   - Speichert Embeddings aller Dokumente
#   - Nutzt spezielle Indexstrukturen für schnelle Suche
#   - Auch große Sammlungen von Dokumenten können schnell durchsucht werden

# %% [markdown]
#
# ## Qdrant
#
# - **Open Source** (Apache 2.0 Lizenz)
# - Einfach zu installieren: `pip install qdrant-client langchain-qdrant`
# - Funktioniert auf Windows und Linux (Linux wird für Produktion empfohlen)
# - Lokaler Modus ohne Server-Konfiguration
# - Unterstützt Hybrid Search (semantische + Keyword-Suche)
# - Gute LangChain-Integration

# %% [markdown]
#
# ## Installation

# %%
# !pip install --root-user-action=ignore --quiet qdrant-client langchain-qdrant

# %%
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# %%
load_dotenv()

# %% [markdown]
#
# ## Qdrant Modi
#
# 1. **In-Memory**: Daten nur im RAM (für Tests und Experimente)
#    - `location=":memory:"`
# 2. **Persistent**: Daten auf Festplatte (für echte Anwendungen)
#    - `path="./qdrant_db"`
#
# Wir verwenden Qdrant über die **LangChain-Integration**

# %% [markdown]
#
# ## Embedding-Modell erstellen
#
# - Qdrant braucht ein **Embedding-Modell**
# - Wir verwenden OpenAI Embeddings über OpenRouter

# %%
embedding_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small",
)

# %%
texts = [
    "Neuronale Netze können komplexe Muster lernen",
    "Gradientenabstieg optimiert Modellparameter",
    "Overfitting tritt auf, wenn Modelle Trainingsdaten auswendig lernen",
    "RAG kombiniert Retrieval mit Generation",
]

# %% [markdown]
#
# ## Vektor-Store erstellen und Dokumente hinzufügen
#
# - `QdrantVectorStore.from_texts()` erstellt den Store und fügt Texte hinzu
# - Embeddings werden **automatisch** berechnet

# %%

# %% [markdown]
#
# ## Dokumente suchen
#
# - `similarity_search()` findet die ähnlichsten Dokumente
# - Parameter `k` bestimmt die Anzahl der Ergebnisse

# %% [markdown]
#
# Suche nach "Was ist Overfitting?":

# %%

# %% [markdown]
#
# Ergebnisse:

# %%

# %% [markdown]
#
# ## Suche mit Ähnlichkeits-Scores
#
# - `similarity_search_with_score()` gibt auch einen Score zurück
# - Der Score ist die **Kosinus-Ähnlichkeit**
#   - Hoher Score = hohe Ähnlichkeit
#
# | Score | Bedeutung |
# |---|---|
# | 0.90+ | Sehr ähnlich |
# | 0.50 | Mittelmäßig ähnlich |
# | 0.10 | Wenig ähnlich |

# %%

# %%

# %% [markdown]
#
# ## Was passiert bei irrelevanten Anfragen?

# %%

# %%

# %% [markdown]
#
# ## Vektor-Datenbanken liefern immer Ergebnisse!
#
# - Auch bei irrelevanten Fragen gibt die Datenbank Ergebnisse zurück
# - Sie wählt einfach die **ähnlichsten** Dokumente — auch wenn keines relevant ist
# - **Niedriger Score** = niedrige Relevanz
# - Wir können nach Ähnlichkeit filtern, um irrelevante Ergebnisse auszuschließen

# %% [markdown]
#
# ## Ergebnisse nach Ähnlichkeit filtern
#
# - Einen Schwellenwert setzen: Nur Ergebnisse mit hoher Ähnlichkeit behalten
# - Schwellenwert hängt von der Anwendung ab (typisch: 0.3–0.5)

# %%
threshold = 0.4

# %% [markdown]
#
# ### Relevante Anfrage

# %%
results_relevant = vectorstore.similarity_search_with_score("Was ist Overfitting?", k=2)

# %%
for doc, score in results_relevant:
    status = "behalten" if score >= threshold else "FILTERN"
    print(f"  [{score:.3f}] ({status}) {doc.page_content}")

# %% [markdown]
#
# ### Irrelevante Anfrage

# %%
results_irrelevant = vectorstore.similarity_search_with_score(
    "Was ist das Rezept für Schokoladenkuchen?", k=2
)

# %%
for doc, score in results_irrelevant:
    status = "behalten" if score >= threshold else "FILTERN"
    print(f"  [{score:.3f}] ({status}) {doc.page_content}")

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Qdrant**: Open-Source Vektor-Datenbank, einfach installierbar ohne Server-Konfiguration
# - **LangChain-Integration**: `QdrantVectorStore` für nahtlose Nutzung
# - **Ähnlichkeitssuche**: `similarity_search()` und `similarity_search_with_score()`
# - **Ähnlichkeits-Scores**: Hoch = ähnlich (Kosinus-Ähnlichkeit)
# - **Achtung**: Liefert immer Ergebnisse — Ähnlichkeits-Filter verwenden!
#
# **Nächster Schritt**: Metadaten und Retriever!
