# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>RAG mit LangChain implementieren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bis jetzt haben
#
# - ✅ LLMs nutzen mit LangChain
# - ✅ Text in Chunks aufteilen
# - ✅ Embeddings erstellen
# - ✅ ChromaDB für Vektor-Speicherung
#
# **Jetzt**: Alles zusammenbauen zu einem RAG-System!

# %%
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# %% [markdown]
#
# ## RAG-Pipeline Übersicht
#
# **Einmalig (Setup)**:
# 1. Dokumente laden
# 2. In Chunks aufteilen
# 3. In ChromaDB speichern (mit Embeddings)
#
# **Bei jeder Anfrage**:
# 4. Anfrage embedden
# 5. Ähnlichste Chunks finden (Retrieval)
# 6. Als Kontext an LLM geben (Augmentation)
# 7. Antwort generieren (Generation)

# %% [markdown]
#
# ## Schritt 1: Dokumente laden und chunken

# %%
# Example documents about machine learning
docs_content = [
    """Linear Regression ist eine Methode des überwachten Lernens.
    Sie modelliert lineare Beziehungen zwischen Features und Zielvariable.
    Die Normalgleichung kann verwendet werden, um optimale Parameter zu finden.""",

    """Neural Networks bestehen aus Schichten von Neuronen.
    Jedes Neuron berechnet eine gewichtete Summe und wendet eine Aktivierungsfunktion an.
    Training erfolgt durch Backpropagation und Gradient Descent.""",

    """Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt.
    Regularisierung und Cross-Validation helfen, Overfitting zu vermeiden.
    Ein gutes Modell generalisiert auf neue, ungesehene Daten.""",

    """Large Language Models sind sehr große neuronale Netze.
    Sie werden auf Milliarden von Wörtern trainiert.
    RAG hilft LLMs, präziser mit spezifischem Wissen zu antworten."""
]

# %%
# Convert to Document objects
documents = [Document(page_content=doc) for doc in docs_content]

# %%

# %% [markdown]
#
# ## Text-Splitter (optional)
#
# Für längere Dokumente: Chunks erstellen

# %%
# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split documents (if needed)
# splits = text_splitter.split_documents(documents)

# %% [markdown]
#
# ## Schritt 2: Vektor-Store erstellen mit LangChain

# %%
# TODO: Create Chroma vector store from documents

# %%

# %% [markdown]
#
# ## Schritt 3: Retriever erstellen

# %%
# TODO: Create retriever from vectorstore

# %% [markdown]
#
# ## Schritt 4: RAG Chain mit LCEL

# %%
# TODO: Create RAG chain with retriever using LCEL

# %%

# %%

# %% [markdown]
#
# ## Schritt 5: RAG-System nutzen

# %%
# TODO: Ask question

# %%
result = qa_chain.invoke(question)

# %%

# %% [markdown]
#
# ## Was passiert hier?
#
# 1. **Frage wird embeddet**: "Was ist Overfitting?" → Vektor
# 2. **Suche in ChromaDB**: Finde ähnlichste Dokumente (Retrieval)
# 3. **Kontext wird erstellt**: Relevante Dokumente + Frage
# 4. **LLM antwortet**: Basierend auf echten Dokumenten (Generation)
# 5. **Rückgabe**: Antwort (`answer`) + Quellen (`context`)
#
# **Alles orchestriert durch LCEL (LangChain Expression Language)!**

# %% [markdown]
#
# ## Weitere Fragen testen

# %%

# %% [markdown]
#
# ## Retrieval-Parameter optimieren
#
# - **k**: Wie viele Dokumente abrufen? (Standard: 4)
# - **search_type**: Wie suchen?
#   - `"similarity"`: Nach Ähnlichkeit (Standard)
#   - `"mmr"`: Maximum Marginal Relevance (diverse Results)
# - **score_threshold**: Nur Dokumente über bestimmter Ähnlichkeit

# %%
# Create retriever with custom parameters
retriever_custom = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Top 3 documents
)

# %% [markdown]
#
# ## Workshop-Vorschau: Eigenes RAG-System
#
# **Im nächsten Workshop bauen Sie**:
# - RAG-System für Ihre eigenen Dokumente
# - Mit Gradio-Interface
# - Mit Konversationsverlauf
# - Mit Quellen-Anzeige
# - Portfolio-tauglich!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **RAG mit LangChain**: Nur wenige Zeilen Code!
# - **Pipeline**: Dokumente → ChromaDB → Retrieval → LLM
# - **LCEL (LangChain Expression Language)**: Komponierbare Chains mit `|` Operator
# - **Automatisch**: Embeddings, Suche, Kontext-Erstellung
# - **Transparent**: Quellen-Dokumente werden zurückgegeben
#
# **Nächster Schritt**: Workshop - Eigenes RAG-System bauen!
