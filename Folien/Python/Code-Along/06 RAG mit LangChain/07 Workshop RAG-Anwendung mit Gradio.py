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
# - Eigene Dokumente hochladen
# - Fragen stellen
# - Antworten mit Quellenangabe
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
# Example document
sample_doc = """
# Machine Learning Kurs - Kapitel 3: Linear Regression

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
"""

# TODO: Create documents and vectorstore

# %%

# %% [markdown]
#
# ## Teil 2: Conversational RAG
#
# RAG + Konversationsverlauf

# %%
# TODO: Create LLM and retriever

# %%
# TODO: Test the retriever and LLM

# %%

# %%

# %%

# %% [markdown]
#
# ## Teil 3: Gradio Interface

# %%
def rag_chatbot(message, history):
    """RAG chatbot function for Gradio"""
    # TODO: Implement RAG query
    pass

# %%
# TODO: Create Gradio interface

# %% [markdown]
#
# ## Teil 4: Dokument-Upload (Erweitert)
#
# Ermögliche Nutzern, eigene Dokumente hochzuladen

# %%
def upload_and_query(file, question):
    """Upload document and answer question"""
    # TODO: Load document, create vectorstore, build LCEL chain, answer question
    pass

# %%
# TODO: Create Gradio interface with file upload

# %% [markdown]
#
# ## Workshop-Aufgaben
#
# ### Basis (Pflicht):
# 1. RAG-System für vorgegebene Dokumente erstellen
# 2. Retriever und LLM mit History-Handling implementieren
# 3. Gradio ChatInterface hinzufügen
# 4. Quellen-Anzeige einbauen
# 5. Verschiedene Fragen testen
#
# ### Erweitert (Optional):
# 6. Dokument-Upload-Funktion
# 7. Mehrere Dokumente gleichzeitig
# 8. PDF-Support
# 9. Retrieval-Parameter anpassen (k, search_type)
# 10. Custom styling

# %% [markdown]
#
# ## Projekt-Ideen
#
# 1. **Persönliche Wissensdatenbank**: Ihre Notizen durchsuchbar
# 2. **Dokumentations-Assistent**: Fragen zu technischen Docs
# 3. **Rezept-Finder**: Fragen zu Kochrezepten
# 4. **Forschungsassistent**: Papers durchsuchen
# 5. **Kurs-Assistent**: Fragen zu Kursmaterialien
#
# **Wählen Sie ein Projekt für Ihr Portfolio!**

# %% [markdown]
#
# ## Zusammenfassung
#
# **Was wir erreicht haben**:
# - ✅ Vollständiges RAG-System mit LangChain
# - ✅ ChromaDB für Vektor-Speicherung
# - ✅ Konversationsverlauf mit Gradio
# - ✅ Gradio-Interface mit Quellenangabe
# - ✅ Portfolio-taugliches Projekt!
#
# **Nächster Schritt**: LLM Orchestration - Noch komplexere Workflows!

# %%
