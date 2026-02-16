# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: RAG-Anwendung mit Gradio</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
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

# %%
import os
import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# %% [markdown]
#
# ## Teil 1: Basis-RAG-System
#
# Zuerst: Einfaches RAG ohne UI

# %%
# Example documents
sample_docs = [
    """# Machine Learning Kurs - Kapitel 3: Linear Regression

Linear Regression ist eine grundlegende Methode des überwachten Lernens.

## Konzept
- Modelliert lineare Beziehungen zwischen Features (X) und Ziel (y)
- Formel: y = w*X + b

## Training
- Normalgleichung für exakte Lösung
- Gradient Descent für große Datensätze
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
]

# TODO: Create Document objects from sample_docs
# TODO: Split documents into chunks using RecursiveCharacterTextSplitter
# TODO: Create embeddings and vectorstore with Chroma

# %%

# %% [markdown]
#
# ## Teil 2: Conversational RAG
#
# RAG + Konversationsverlauf

# %%
# TODO: Create LLM (using OpenRouter) and retriever

# %%
# TODO: Test the retriever and LLM with a question

# %%

# %%

# %%

# %% [markdown]
#
# ## Teil 3: Gradio Interface

# %%
def rag_chatbot(message, history):
    """RAG chatbot function for Gradio"""
    # TODO: Implement RAG query with retriever, context, and history
    pass

# %%
# TODO: Create Gradio ChatInterface

# %% [markdown]
#
# ## Teil 4: Dokument-Upload (Erweitert)
#
# Ermögliche Nutzern, eigene Dokumente hochzuladen

# %%
def upload_and_query(file, question):
    """Upload document and answer question"""
    # TODO: Read file, split into chunks, create vectorstore, build RAG chain, answer
    pass

# %%
# TODO: Create Gradio interface with file upload

# %% [markdown]
#
# ## Zusammenfassung
#
# **Was wir erreicht haben**:
# - ✅ Vollständiges RAG-System mit LangChain
# - ✅ Text-Chunking für bessere Retrieval-Qualität
# - ✅ ChromaDB für Vektor-Speicherung
# - ✅ Konversationsverlauf mit Gradio
# - ✅ Gradio-Interface mit Quellenangabe
# - ✅ Portfolio-taugliches Projekt!
#
# **Nächster Schritt**: LLM Orchestration — Noch komplexere Workflows!

# %% [markdown]
#
# ## Workshop-Aufgaben
#
# ### Basis (Pflicht):
# 1. Erstellen Sie ein RAG-System für die vorgegebenen Dokumente
# 2. Implementieren Sie einen Retriever mit Konversationsverlauf
# 3. Fügen Sie ein Gradio ChatInterface hinzu
# 4. Bauen Sie eine Quellen-Anzeige in die Antworten ein
# 5. Testen Sie das System mit verschiedenen Fragen
#
# ### Erweitert (Optional):
# 6. Implementieren Sie die Dokument-Upload-Funktion
# 7. Ermöglichen Sie das Laden mehrerer Dokumente gleichzeitig
# 8. Fügen Sie PDF-Support hinzu (mit PyPDFLoader)
# 9. Experimentieren Sie mit Retrieval-Parametern (k, search_type)
# 10. Gestalten Sie das Interface mit Custom Styling

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
