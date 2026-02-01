# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LangChain: Ein einfacher Chatbot</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bisher gebaut haben
#
# Mit `requests`/OpenAI API haben wir einen funktionierenden Chatbot erstellt:
#
# - `SimpleChatbot` Klasse mit `self.messages`
# - Konversationsverlauf wird gespeichert
# - System-Prompt für Persönlichkeit
#
# **Das funktioniert gut!** Aber...

# %% [markdown]
#
# ## Verschiedene APIs
#
# Viele Anbieter nutzen OpenAI-kompatible APIs:
#
# - OpenRouter, Together AI, Groq, lokale Server (Ollama, vLLM)
# - Gleicher Code funktioniert mit verschiedenen Anbietern
#
# Aber nicht alle:
#
# - **Anthropic Claude** hat eine eigene API
# - Manche Anbieter haben spezielle Funktionen
#
# Außerdem: Direkte API-Aufrufe erfordern viel Boilerplate-Code

# %% [markdown]
#
# ## Die Lösung: LangChain
#
# - **Framework** für LLM-Anwendungen
# - **Einheitliche Schnittstelle** für alle Anbieter
# - Gleicher Code funktioniert mit OpenAI, Anthropic, lokalen Modellen...
# - **Industrie-Standard** (in 60%+ der Job-Anzeigen!)

# %% [markdown]
#
# ## Installation

# %%
# !pip install langchain langchain-core langchain-openai langchain-anthropic

# %%
import os
from dotenv import load_dotenv

# %%
load_dotenv()

# %% [markdown]
#
# ## Der einfachste LangChain-Aufruf
#
# Nur zwei Zeilen Code für OpenAI:

# %%
from langchain_openai import ChatOpenAI

# %%

# %%

# %%

# %% [markdown]
#
# ## Das Ergebnis verstehen
#
# - `llm.invoke()` gibt ein `AIMessage` Objekt zurück
# - `.content` enthält den Text der Antwort
# - Viel einfacher als mit `requests`!
# - `OPENAI_API_KEY` wird automatisch aus der Umgebung gelesen

# %% [markdown]
#
# ## Mit OpenRouter
#
# Für OpenRouter geben wir API-Key und Base-URL an:

# %%
model = "mistralai/ministral-14b-2512"

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model,
)

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Nachrichtentypen in LangChain
#
# LangChain verwendet spezielle Klassen für Nachrichten:
#
# - `HumanMessage`: Nachricht vom Benutzer
# - `AIMessage`: Antwort vom LLM
# - `SystemMessage`: System-Prompt (Anweisungen)

# %%
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# %% [markdown]
#
# ## Nachrichten als Liste
#
# Genau wie bei der REST API können wir eine Liste von Nachrichten senden:

# %%

# %%

# %%

# %% [markdown]
#
# ## Unser LangChain-Chatbot
#
# Jetzt bauen wir einen Chatbot mit LangChain!
#
# Die Struktur ist **identisch** zu unserem `SimpleChatbot`:
# - `self.messages` speichert den Verlauf
# - `chat()` Methode für Konversation
#
# Aber: **Viel weniger Code!**

# %%

# %% [markdown]
#
# ## Den Chatbot testen

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Der Verlauf
#
# Unser Chatbot erinnert sich an alles:

# %%
for msg in bot.messages:
    role = type(msg).__name__
    content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
    print(f"{role}: {content}")

# %% [markdown]
#
# ## Vorteil: Provider wechseln
#
# Mit einer kleinen Änderung können wir den Provider wechseln.

# %%
from langchain_core.language_models.base import BaseLanguageModel
from langchain_anthropic import ChatAnthropic


# %%
def create_llm(provider) -> BaseLanguageModel:
    """Create LLM based on provider name."""
    if provider == "openrouter":
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model,
        )
    elif provider == "openai":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-5.2",
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-haiku-4-5",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

# %%
class FlexibleChatbot:
    """A chatbot that can use different providers."""

    def __init__(self, provider="openrouter", system_prompt=None):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model,
        )
        self.messages = []
        if system_prompt:
            self.messages.append(SystemMessage(content=system_prompt))

    def chat(self, user_message):
        """Send a message and get a response."""
        self.messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(self.messages)
        self.messages.append(response)
        return response.content



# %% [markdown]
#
# ## Verschiedene Provider testen
#
# Gleicher Code, verschiedene Modelle:

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Alternative: RunnableWithMessageHistory
#
# LangChain bietet auch `RunnableWithMessageHistory` für Session-basierte
# Konversationen:
#
# - Automatische Verwaltung des Nachrichtenverlaufs
# - Unterstützung für mehrere Sessions (z.B. verschiedene Benutzer)
# - Integration mit verschiedenen Speicher-Backends

# %%
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# %% [markdown]
#
# ### Aufbau eines Chatbots mit RunnableWithMessageHistory

# %%
store = {}

# %%
def get_session_history(session_id) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# %%
chatbot_with_history = RunnableWithMessageHistory(llm, get_session_history)

# %%
session_id = "user_123"

# %%

# %%

# %% [markdown]
#
# ### Der Verlauf wird automatisch verwaltet

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **LangChain** abstrahiert verschiedene LLM-Anbieter
# - **Gleicher Code** funktioniert mit OpenAI, Anthropic, etc.
# - **Klassen-basierter Chatbot** mit `self.messages` (wie vorher)
# - **Nachrichtentypen**: `HumanMessage`, `AIMessage`, `SystemMessage`
# - **Provider-Wechsel** mit minimalen Änderungen

# %% [markdown]
#
# ## Gradio-Chatbot mit System-Prompt
#
# Vorbereitung für den Workshop:
# - Wir bauen einen Chatbot mit wählbarem System-Prompt
#   - Im Notebook oder als eigenständige App: `Projekte/SimpleLangChainChatbot`

# %% [markdown]
#
# ## Wie `additional_inputs` funktioniert
#
# `gr.ChatInterface` ruft unsere Funktion `fn` mit folgenden Parametern auf:
#
# 1. `message` - Die aktuelle Benutzernachricht
# 2. `history` - Die bisherige Konversation (Liste von Nachrichten)
# 3. **Weitere Parameter** aus `additional_inputs` (in der gleichen Reihenfolge!)

# %% [markdown]
#
# ### Beispiel
#
# ```python
# additional_inputs=[
#     gr.Dropdown(...),  # → 3. Parameter
#     gr.Slider(...),    # → 4. Parameter
# ]
#
# def fn(message, history, dropdown_value, slider_value):
#     ...
# ```
#
# Die Reihenfolge in `additional_inputs` bestimmt die Reihenfolge der Parameter!

# %%
import gradio as gr

# %%
SYSTEM_PROMPTS = {
    "Helpful Assistant": "You are a helpful assistant.",
    "Python Tutor": "You are a friendly Python tutor who explains concepts simply.",
    "Pirate": "You are a pirate. Answer all questions like a pirate would.",
}

# %%
def chat_with_system_prompt(message, history):
    """Chatbot that uses the selected system prompt."""
    system_prompt = "You are a helpful assistant."

    messages: list = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=message))

    response = llm.invoke(messages)
    return response.content

# %%
system_prompt_demo = gr.ChatInterface(
    fn=chat_with_system_prompt,
    additional_inputs=[
        gr.Dropdown(
            choices=list(SYSTEM_PROMPTS.keys()),
            value="Helpful Assistant",
            label="System Prompt",
        ),
    ],
    title="Chatbot mit System-Prompt",
    description="Wählen Sie einen System-Prompt!",
)

# %%

# %% [markdown]
#
# ## Workshop: Multi-Provider Chatbot
#
# **Ziel**: Einen Chatbot bauen, bei dem man den Provider wählen kann!
#
# **Aufgaben**:
# 1. Erstellen Sie einen `FlexibleChatbot` mit Gradio-Interface
# 2. Fügen Sie ein Dropdown hinzu, um den Provider zu wählen
# 3. Testen Sie verschiedene Provider

# %% [markdown]
#
# ### Teil 1: Chatbot-Funktion
#
# Wir brauchen eine Funktion, die mit Gradio funktioniert:

# %%
chatbot_instances = {}


# %%
def multi_provider_chat(message, history, provider):
    """Chatbot that can switch providers."""
    # TODO: Implement chatbot that uses FlexibleChatbot
    # Hint: Store chatbot instance in chatbot_instances dict
    pass

# %% [markdown]
#
# ### Teil 2: Gradio Interface

# %%
# TODO: Create ChatInterface with provider dropdown

# %%
