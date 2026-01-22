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
# ## Ein etwas komplizierterer Generator


# %%

# %%
text = """\
# Introduction to Python Lists

Python lists are a fundamental data structure used to store a collection of \
items. They are ordered, mutable, and allow duplicate elements. Lists are \
defined by enclosing sequence of items in square brackets `[]`, separated by \
commas.

## Creating a List

You can create a list with various data types, including integers, floats, \
strings, and even other lists. Here are a few examples:
"""

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
# ## Ein einfacher Chatbot
#
# Zuerst erstellen wir einen einfachen Chatbot ohne Streaming.
#
# Diesmal speichern wir die Listen der Nachrichten nicht im Chatbot-Objekt,
# sondern verwenden die History von Gradio.

# %%
import gradio as gr


# %%
def create_llm():
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/ministral-14b-2512",
    )


# %% [markdown]
#
# ## Warum Gradios History verwenden?
#
# Gradio bietet Funktionen wie **Nachricht löschen** und **Antwort neu generieren**
#
# - Damit diese korrekt funktionieren, müssen wir Gradios History verwenden
# - Wir speichern die Nachrichten nicht mehr im Chatbot-Objekt
# - `history_to_messages()` konvertiert zwischen Gradio- und LangChain-Format

# %%
def history_to_messages(history, system_prompt=None):
    """Convert Gradio history to LangChain messages."""
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

# %%
class SimpleChatbot:
    """A simple chatbot using OpenRouter."""

    def __init__(self, system_prompt=None):
        self.llm = create_llm()
        self.system_prompt = system_prompt

    def chat(self, user_message, history):
        """Send a message and get a response."""
        messages = history_to_messages(history, self.system_prompt)
        messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(messages)
        return response.content


# %% [markdown]
#
# ## Nicht-Streaming Chatbot mit Gradio
#
# Beobachten Sie: Die Antwort erscheint erst nach einer Wartezeit!

# %%
simple_bot = SimpleChatbot("Du bist ein hilfreicher Assistent.")


# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Streaming zum Chatbot hinzufügen
#
# Wir erweitern unseren `SimpleChatbot` mit einer `stream()`-Methode:

# %%
class SimpleChatbot:
    def __init__(self, system_prompt=None):
        self.llm = create_llm()
        self.system_prompt = system_prompt

    def chat(self, user_message, history):
        """Send a message and get a response."""
        messages = history_to_messages(history, self.system_prompt)
        messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(messages)
        return response.content


# %% [markdown]
#
# ## Streaming Chatbot mit Gradio
#
# Jetzt sehen wir die Antwort Wort für Wort entstehen!

# %%
streaming_bot = SimpleChatbot("Du bist ein hilfreicher Assistent.")

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Workshop: Streaming-Chatbot erweitern
#
# **Aufgabe**: Erweitern Sie den Streaming-Chatbot!
#
# 1. Fügen Sie eine Provider-Auswahl hinzu (openrouter, openai, anthropic)
# 2. Fügen Sie eine System-Prompt-Auswahl hinzu
# 3. Der Benutzer soll sowohl Provider als auch System-Prompt wählen können
# 4. **Bonus**: Zeigen Sie an, wie viele Zeichen bereits generiert wurden

# %%
SYSTEM_PROMPTS = {
    "Helpful Assistant": "You are a helpful assistant.",
    "Python Tutor": "You are a friendly Python tutor who explains concepts simply.",
    "Pirate": "You are a pirate. Answer all questions like a pirate would.",
}


# %%
from langchain_anthropic import ChatAnthropic


# %%

# %%
def extended_streaming_chat(message, history, provider, system_prompt_name):
    """Streaming chatbot with provider and system prompt selection."""
    # TODO: Implementieren Sie den Streaming-Chatbot
    # Hint: Erstellen Sie für jede Kombination aus Provider und System-Prompt
    #       eine eigene Chatbot-Instanz
    pass

# %%
# TODO: Erstellen Sie das ChatInterface mit beiden Dropdowns

# %%

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
