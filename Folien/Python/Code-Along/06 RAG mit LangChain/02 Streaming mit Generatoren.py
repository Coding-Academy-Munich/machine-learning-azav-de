# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Streaming mit Generatoren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Warum Streaming?
#
# **Problem:** LLM-Antworten können lang sein
#
# - Der Benutzer wartet... und wartet...
# - Erst am Ende erscheint die komplette Antwort
# - Das fühlt sich langsam an!

# %% [markdown]
#
# ## Die Lösung: Streaming
#
# - Zeige die Antwort **während sie generiert wird**
# - Wie ein Brief, den man Wort für Wort liest
# - Der Benutzer sieht sofort, dass etwas passiert

# %% [markdown]
#
# ## Das Schlüsselwort `yield`
#
# Normale Funktionen mit `return`:
#
# - Funktion gibt **einen** Wert zurück
# - Funktion **endet** sofort

# %%
def normal_function():
    return "Erster Wert"
    return "Zweiter Wert"  # Wird nie erreicht!

# %%

# %% [markdown]
#
# ## Funktionen mit `yield`
#
# - Funktion gibt einen Wert zurück und **pausiert**
# - Bei der nächsten Anfrage macht sie **weiter**
# - Kann **mehrere Werte** nacheinander liefern

# %%
def generator_function():
    yield "Erster Wert"
    yield "Zweiter Wert"
    yield "Dritter Wert"

# %% [markdown]
#
# ## Generator-Objekte
#
# Ein Aufruf einer Generator-Funktion gibt ein **Generator-Objekt** zurück:

# %%

# %%

# %% [markdown]
#
# ## Werte aus einem Generator holen
#
# Mit einer `for`-Schleife können wir alle Werte durchlaufen:

# %%

# %% [markdown]
#
# ## Generatoren in Gradio
#
# Gradio unterstützt Generator-Funktionen automatisch!
#
# - Bei jedem `yield` wird die Chat-Oberfläche aktualisiert
# - Der Benutzer sieht die Antwort wachsen

# %% [markdown]
#
# ## Das Muster für Streaming
#
# ```python
# def streaming_chat(message, history):
#     full_response = ""
#     for chunk in ...:  # Chunks vom LLM
#         full_response += chunk
#         yield full_response  # Zeige bisherige Antwort
# ```
#
# Wichtig: Wir geben immer die **gesamte bisherige Antwort** zurück!

# %% [markdown]
#
# ## Streaming mit LangChain
#
# LangChain bietet `llm.stream()` statt `llm.invoke()`:
#
# - Gibt Chunks zurück (kleine Teile der Antwort)
# - Jeder Chunk hat `.content` mit dem Text

# %%
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# %%
load_dotenv()

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
)

# %% [markdown]
#
# ## Beispiel: Streaming in Aktion

# %%

# %% [markdown]
#
# ## Streaming-Chatbot mit Gradio
#
# Jetzt kombinieren wir alles:

# %%
import gradio as gr
from langchain_anthropic import ChatAnthropic

# %%
chatbot_instances = {}


# %%
class FlexibleChatbot:
    """A chatbot that can use different providers."""

    def __init__(self, provider="openrouter", system_prompt=None):
        self.llm = self._create_llm(provider)
        self.messages = []
        if system_prompt:
            self.messages.append(SystemMessage(content=system_prompt))

    def _create_llm(self, provider):
        """Create LLM based on provider name."""
        if provider == "openrouter":
            return ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model="mistralai/ministral-14b-2512",
            )
        elif provider == "openai":
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model="claude-haiku-4-5",
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def chat(self, user_message):
        """Send a message and get a response."""
        self.messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(self.messages)
        self.messages.append(response)
        return response.content


# %%
def streaming_chat(message, _history, provider):
    """Chatbot with streaming responses."""
    global chatbot_instances

    if provider not in chatbot_instances:
        chatbot_instances[provider] = FlexibleChatbot(
            provider, "Du bist ein hilfreicher Assistent."
        )

    bot = chatbot_instances[provider]
    bot.messages.append(HumanMessage(content=message))

    full_response = ""
    for chunk in bot.llm.stream(bot.messages):
        if chunk.content:
            full_response += chunk.content
            yield full_response

    bot.messages.append(AIMessage(content=full_response))


# %%

# %%

# %% [markdown]
#
# ## Workshop: Streaming-Chatbot erweitern
#
# **Aufgabe**: Erweitern Sie den Streaming-Chatbot!
#
# 1. Fügen Sie einen System-Prompt-Auswahl hinzu (wie im vorherigen Workshop)
# 2. Der Benutzer soll sowohl Provider als auch System-Prompt wählen können
# 3. **Bonus**: Zeigen Sie an, wie viele Zeichen bereits generiert wurden

# %%
SYSTEM_PROMPTS = {
    "Helpful Assistant": "You are a helpful assistant.",
    "Python Tutor": "You are a friendly Python tutor who explains concepts simply.",
    "Pirate": "You are a pirate. Answer all questions like a pirate would.",
}


# %%
def extended_streaming_chat(message, _history, provider, system_prompt_name):
    """Streaming chatbot with provider and system prompt selection."""
    # TODO: Implementieren Sie den Streaming-Chatbot
    # Hint: Erstellen Sie für jede Kombination aus Provider und System-Prompt
    #       eine eigene Chatbot-Instanz
    pass

# %%
# TODO: Erstellen Sie das ChatInterface mit beiden Dropdowns

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **`yield`** pausiert eine Funktion und gibt einen Wert zurück
# - **Generator-Funktionen** können mehrere Werte nacheinander liefern
# - **Gradio** aktualisiert die UI bei jedem `yield`
# - **`llm.stream()`** liefert die Antwort in kleinen Chunks
# - **Streaming** macht Chatbots reaktiver und benutzerfreundlicher
