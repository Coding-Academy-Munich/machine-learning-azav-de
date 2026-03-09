# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: RAG-Anwendung mit Gradio</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Workshop-Ziel
#
# Wir bauen ein vollständiges RAG-System:
# - Eigene Dokumente laden und chunken
# - Fragen stellen und Antworten mit Quellenangabe erhalten
# - Professionelles Gradio-Interface
# - Konversationsverlauf

# %% [markdown]
#
# ## Workshop-Aufgaben: Basis (Pflicht)
#
# 1. Erstellen Sie ein RAG-System für die vorgegebenen Dokumente
# 2. Implementieren Sie einen Retriever mit Konversationsverlauf
# 3. Fügen Sie ein Gradio ChatInterface hinzu
# 4. Bauen Sie eine Quellen-Anzeige in die Antworten ein
# 5. Testen Sie das System mit verschiedenen Fragen

# %% [markdown]
#
# ## Workshop-Aufgaben: Erweitert (Optional)
#
# 6. Implementieren Sie die Dokument-Upload-Funktion
# 7. Ermöglichen Sie das Laden mehrerer Dokumente gleichzeitig
# 8. Fügen Sie PDF-Support hinzu (mit PyPDFLoader)
# 9. Experimentieren Sie mit Retrieval-Parametern (k, search_type)
# 10. Gestalten Sie das Interface mit Custom Styling

# %%
# ! pip install qdrant-client langchain-qdrant

# %%
import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# %%
load_dotenv()

# %%
sample_docs = [
    """# Machine Learning Kurs - Kapitel 3: Linear Regression

Linear Regression ist eine grundlegende Methode des überwachten Lernens.

## Konzept
- Modelliert lineare Beziehungen zwischen Features (X) und Ziel (y)
- Formel: y = w*X + b

## Training
- Normalgleichung für exakte Lösung
- Gradient Descent für große Datensätze

## Evaluation
- Mean Squared Error (MSE) als Loss-Funktion
- R² Score für Modellqualität
""",
    """# Machine Learning Kurs - Kapitel 4: Neural Networks

Neural Networks sind inspiriert von biologischen Neuronen.

## Aufbau
- Input Layer, Hidden Layers, Output Layer
- Jedes Neuron: gewichtete Summe + Aktivierungsfunktion

## Training
- Backpropagation berechnet Gradienten
- Gradient Descent passt Gewichte an
""",
    """# Machine Learning Kurs - Kapitel 5: Overfitting

Overfitting ist ein häufiges Problem beim Machine Learning.

## Symptome
- Sehr gute Performance auf Trainingsdaten
- Schlechte Performance auf Testdaten

## Lösungen
- Mehr Trainingsdaten sammeln
- Regularisierung (L1, L2)
- Cross-Validation
- Early Stopping
"""
]

# %%
documents = [Document(page_content=doc) for doc in sample_docs]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# %%
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
    temperature=0
)

# %% [markdown]
#
# ## Teil 1: Basis-RAG-System
#
# Erstellen Sie einen Vektor-Store aus den vorgegebenen Dokumenten und
# testen Sie den Retriever mit einer Frage.

# %%

# %% [markdown]
# Vectorstore erstellt — Anzahl Chunks:

# %%

# %%

# %%
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# %% [markdown]
#
# ## Teil 2: Conversational RAG
#
# Verwenden Sie den Retriever, um relevante Dokumente zu finden, und
# senden Sie den Kontext zusammen mit der Frage an das LLM.

# %%

# %% [markdown]
# Frage:

# %%

# %% [markdown]
# Gefundene Dokumente:

# %%

# %%

# %% [markdown]
# Antwort:

# %%

# %% [markdown]
#
# ## Teil 3: Gradio Interface
#
# Erstellen Sie eine `rag_chatbot`-Funktion, die Retrieval, Kontext-Aufbau,
# Konversationsverlauf und Quellenangabe kombiniert. Binden Sie sie in ein
# Gradio `ChatInterface` ein.

# %%

# %%

# %% [markdown]
#
# ## Teil 4: Dokument-Upload (Erweitert)
#
# Erstellen Sie eine `upload_and_query`-Funktion, die eine hochgeladene Datei
# liest, in Chunks aufteilt, einen temporären Vektor-Store erstellt und eine
# RAG Chain darauf ausführt. Binden Sie sie in ein Gradio `Interface` ein.

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# **Was wir erreicht haben**:
# - Vollständiges RAG-System mit LangChain
# - Text-Chunking für bessere Retrieval-Qualität
# - Qdrant mit Hybrid Search für Vektor-Speicherung
# - Gradio-Interface mit Konversationsverlauf und Quellenangabe
# - Portfolio-taugliches Projekt!
#
# **Nächster Schritt**: Wie wissen wir, ob unser RAG-System gut funktioniert? Evaluierung!

# %% [markdown]
#
# ## Projekt-Ideen
#
# 1. **Persönliche Wissensdatenbank**: Ihre Notizen durchsuchbar machen
# 2. **Dokumentations-Assistent**: Fragen zu technischen Dokumentationen beantworten
# 3. **Rezept-Finder**: Fragen zu Kochrezepten stellen
# 4. **Forschungsassistent**: Papers und Artikel durchsuchen
# 5. **Kurs-Assistent**: Fragen zu Kursmaterialien beantworten
#
# **Wählen Sie ein Projekt für Ihr Portfolio!**
