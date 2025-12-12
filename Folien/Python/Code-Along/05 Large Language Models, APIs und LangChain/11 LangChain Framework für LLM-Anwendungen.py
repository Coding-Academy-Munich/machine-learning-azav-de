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
# Create LLM (using OpenRouter for access to many models)
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512"
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
# ## Workshop: Professioneller Chatbot mit LangChain und Gradio
#
# **Ziel**: Einen vollständigen Chatbot mit professioneller Oberfläche bauen!
#
# Wir kombinieren:
# - LangChain für LLM-Integration
# - Gradio für die Benutzeroberfläche
# - Einstellbare Persönlichkeit
# - Temperatur-Kontrolle

# %%
import gradio as gr


# %% [markdown]
#
# ### Teil 1: Basis-Chatbot mit LangChain
#
# Erstellen wir zunächst einen einfachen Chatbot mit LangChain

# %%
# TODO: Create LangChain chat model
# workshop_llm = ChatOpenAI(...)

# %%
def simple_chatbot(message, history):
    """Simple chatbot function for Gradio"""
    # TODO: Call LLM and return response
    pass

# %% [markdown]
#
# ### Teil 2: Gradio Interface erstellen

# %%
# TODO: Create ChatInterface
# demo = gr.ChatInterface(...)

# %%

# %% [markdown]
#
# ### Teil 3: Persönlichkeit hinzufügen
#
# - System-Prompt definiert die Persönlichkeit
# - Verschiedene Persönlichkeiten zur Auswahl

# %%
PERSONALITIES_DE = {
    "Freundlich": "Du bist ein freundlicher und hilfsbereiter Assistent.",
    "Professionell": "Du bist ein professioneller und sachlicher Assistent.",
    "Lustig": "Du bist ein humorvoller Assistent, der gerne Witze macht.",
}

PERSONALITIES_EN = {
    "Friendly": "You are a friendly and helpful assistant.",
    "Professional": "You are a professional and matter-of-fact assistant.",
    "Funny": "You are a humorous assistant who likes to make jokes.",
}

# %%
PERSONALITIES = PERSONALITIES_DE
DEFAULT_PERSONALITY = "Freundlich"


# %%
def chatbot_with_personality(message, history, personality):
    """Chatbot with configurable personality"""
    # TODO: Use personality in system prompt
    pass

# %% [markdown]
#
# Chat Interface mit Persönlichkeit

# %%
# TODO: Create ChatInterface with personality dropdown

# %%

# %% [markdown]
#
# ### Teil 4: Temperatur-Kontrolle

# %%
def chatbot_full(message, history, personality, temperature):
    """Chatbot with all controls"""
    # TODO: Implement with temperature control
    pass

# %% [markdown]
#
# ### Teil 5: Vollständiges Gradio Interface

# %%
# TODO: Create ChatInterface with all inputs

# %%

# %% [markdown]
#
# ### Teil 6: Zusatzfunktionen (Optional)
#
# **Weitere Features, die Sie hinzufügen können**:
# - Export-Button für Konversation
# - Token-Zähler und Kosten-Tracker
# - Verschiedene Modelle zur Auswahl
# - Custom Styling/Theme
# - Beispiel-Prompts (Buttons)

# %% [markdown]
#
# ### Beispiel-Anwendung: Kundenservice-Bot
#
# **Szenario**: Chatbot für einen Online-Shop
# - Beantwortet Fragen zu Produkten
# - Hilft bei Bestellungen
# - Professionelles Interface

# %%
CUSTOMER_SERVICE_PROMPT_DE = """\
Du bist ein freundlicher Kundenservice-Mitarbeiter für einen Online-Shop.
Du hilfst Kunden bei Fragen zu Produkten, Bestellungen und Versand.
Sei höflich, hilfsbereit und professionell."""


# %%
def customer_service_bot(message, history, temperature):
    """Customer service chatbot"""
    try:
        llm_cs = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="mistralai/ministral-14b-2512",
            temperature=temperature
        )

        messages = [("system", CUSTOMER_SERVICE_PROMPT_DE)]

        # Convert history to LangChain format
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=message))

        response = llm_cs.invoke(messages)
        return response.content

    except Exception as e:
        return f"Entschuldigung, es gab einen Fehler: {str(e)}"

# %%

# %%

# %% [markdown]
#
# ## Workshop-Aufgaben
#
# ### Basis (Pflicht):
# 1. Chatbot mit LangChain erstellen
# 2. Gradio ChatInterface hinzufügen
# 3. Persönlichkeits-Dropdown implementieren
# 4. Temperatur-Slider implementieren
# 5. Fehlerbehandlung testen
#
# ### Erweitert (Optional):
# 6. Export-Funktion hinzufügen
# 7. Token-Counter einbauen
# 8. Custom Theme gestalten
# 9. Beispiel-Prompts als Buttons

# %% [markdown]
#
# ## Zusammenfassung
#
# - **LangChain**: Framework für LLM-Anwendungen
# - **Komponenten**: LLMs, Prompts, Chains (LCEL), Memory, Loaders
# - **Vorteile**: Weniger Code, mehr Funktionalität
# - **Industrie-Standard**: Wichtig für Jobmarkt
# - **Workshop**: Vollständiger Chatbot mit Gradio
#
# **Nächster Schritt**: RAG-Systeme - Dokumente mit LLMs verbinden!

# %%
