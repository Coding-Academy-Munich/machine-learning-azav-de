# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LangChain: Framework für LLM-Anwendungen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bisher gemacht haben
#
# - LLM APIs direkt aufgerufen
# - Konversationsverlauf manuell verwaltet
# - Jede Funktion selbst geschrieben
#
# **Das funktioniert**, aber...
# - Viel Code für Standard-Aufgaben
# - Fehleranfällig
# - Schwer zu erweitern

# %% [markdown]
#
# ## Die Lösung: LangChain
#
# - **Framework** für LLM-Anwendungen
# - **Vorgefertigte Komponenten** für häufige Aufgaben
# - **Weniger Code**, mehr Funktionalität
# - **Industrie-Standard** (in 60%+ der Job-Anzeigen!)
#
# **Installation**: `pip install langchain langchain-core langchain-openai langchain-anthropic`

# %%
# !pip install langchain langchain-core langchain-openai langchain-anthropic

# %%
import os

# %%
from langchain_openai import ChatOpenAI

# %%
from langchain_anthropic import ChatAnthropic

# %%
from langchain_core.prompts import ChatPromptTemplate

# %%
from langchain_core.messages import HumanMessage, AIMessage

# %% [markdown]
#
# ## LangChain Komponenten
#
# 1. **LLMs und Chat Models**: Wrapper für verschiedene Anbieter
# 2. **Prompts**: Template-System für Prompts
# 3. **Chains**: Kombination mehrerer Schritte
# 4. **Memory**: Konversationsverlauf automatisch
# 5. **Document Loaders**: Daten aus verschiedenen Quellen laden
# 6. **Vector Stores**: Integration mit Vektor-Datenbanken

# %% [markdown]
#
# ## Einfacher LLM-Aufruf mit LangChain

# %%
# Create LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

# Call it
# TODO: Call llm.invoke()

# %%

# %%

# %% [markdown]
#
# ## Prompt Templates
#
# - **Problem**: Prompts mit Variablen sind umständlich mit String-Formatierung
# - **Lösung**: LangChain Prompt Templates
# - Wiederverwendbar, klar strukturiert

# %%
from langchain_core.prompts import ChatPromptTemplate

# %% [markdown]
#
# Erzeugen der Template

# %%
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {subject} teacher."),
    ("human", "{question}")
])

# %% [markdown]
#
# Formatieren des Prompts

# %%

# %% [markdown]
#
# Aufrufen des LLMs

# %%

# %% [markdown]
#
# Anzeigen des Ergebnisses

# %%

# %% [markdown]
#
# ## Chains: Schritte verketten
#
# - **Chain**: Verkettung mehrerer Komponenten
# - **Beispiel**: Prompt Template → LLM → Verarbeitung
# - Mit `|` Operator verknüpfen (LCEL - LangChain Expression Language)

# %%
# Create chain
chain = template | llm

# Use chain
# TODO: Invoke chain with inputs

# %%

# %% [markdown]
#
# ## Memory: Konversationsverlauf
#
# - LangChain kann Konversationsverlauf mit **RunnableWithMessageHistory** verwalten
# - Speichert Nachrichten in einer Session
# - Ermöglicht Zugriff auf Konversationskontext
# - Einfacher als manuelle Verwaltung!

# %%
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Document Loaders
#
# - Daten aus verschiedenen Quellen laden
# - **PDFs**, **Webseiten**, **Word-Dokumente**, etc.
# - Mit wenigen Zeilen Code

# %%
from langchain_community.document_loaders import TextLoader

# Example: Load text file (if it exists)
# loader = TextLoader("example.txt")
# documents = loader.load()

# %% [markdown]
#
# ## Vorher vs. Nachher: Chatbot-Code

# %% [markdown]
#
# ### Vorher (Ohne LangChain):
# - API-Client initialisieren
# - Nachrichtenverlauf manuell verwalten
# - Fehlerbehandlung selbst schreiben
# - ~50-80 Zeilen Code
#
# ### Nachher (Mit LangChain):
# - LangChain Chat Model
# - RunnableWithMessageHistory für Memory
# - Automatische Fehlerbehandlung
# - ~15-20 Zeilen Code

# %% [markdown]
#
# ## Zusammenfassung
#
# - **LangChain**: Framework für LLM-Anwendungen
# - **Komponenten**: LLMs, Prompts, Chains (LCEL), Memory, Loaders
# - **Vorteile**: Weniger Code, mehr Funktionalität
# - **Industrie-Standard**: Wichtig für Jobmarkt
#
# **Nächster Schritt**: Text-Verarbeitung für RAG-Systeme!

# %% [markdown]
#
# ## Workshop: Chatbot mit LangChain
#
# **Aufgabe**: Chatbot aus vorheriger Lektion umschreiben
#
# 1. LangChain Chat Model erstellen
# 2. Memory hinzufügen
# 3. Mit Gradio-Interface verbinden
# 4. Vergleichen: Vorher vs. Nachher
#
# **Ziel**: Sehen, wie viel einfacher es mit Frameworks ist!

# %%
