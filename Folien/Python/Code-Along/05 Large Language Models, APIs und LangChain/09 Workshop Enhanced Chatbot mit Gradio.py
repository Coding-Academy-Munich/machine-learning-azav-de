# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: Enhanced Chatbot mit Gradio</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Workshop-Ziel
#
# Einen professionellen Chatbot bauen mit:
# - LangChain für LLM-Integration
# - Gradio für die Benutzeroberfläche
# - Konversationsverlauf (automatisch durch Gradio verwaltet)
# - Einstellbare Persönlichkeit
# - Temperatur-Kontrolle
# - Fehlerbehandlung

# %%
import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# %% [markdown]
#
# ## Teil 1: Basis-Chatbot mit LangChain
#
# Erstellen wir zunächst einen einfachen Chatbot mit LangChain

# %%
# TODO: Create LangChain chat model
# llm = ChatOpenAI(...)

# %%
def simple_chatbot(message, history):
    """Simple chatbot function for Gradio"""
    # TODO: Call LLM and return response
    pass

# %% [markdown]
#
# ## Teil 2: Gradio Interface erstellen

# %%


# %% [markdown]
#
# Starten des Interfaces

# %%

# %% [markdown]
#
# ## Teil 3: Persönlichkeit hinzufügen
#
# - System-Prompt definiert die Persönlichkeit
# - Verschiedene Persönlichkeiten zur Auswahl

# %%
PERSONALITIES_DE = {
    "Freundlich": "Du bist ein freundlicher und hilfsbereiter Assistent.",
    "Professionell": "Du bist ein professioneller und sachlicher Assistent.",
    "Lustig": "Du bist ein humorvoller Assistent, der gerne Witze macht.",
}

# %%
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
    try:
        # Call LLM
        response = llm.invoke(message)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# %% [markdown]
#
# Chat Interface mit Persönlichkeit

# %%

# %%


# %% [markdown]
#
# ## Teil 4: Temperatur-Kontrolle

# %%
def chatbot_full(message, history, personality, temperature):
    """Chatbot with all controls"""
    # TODO: Implement with temperature control
    try:
        # Get system prompt
        system_prompt = PERSONALITIES.get(personality, PERSONALITIES[DEFAULT_PERSONALITY])

        # Create messages with system prompt
        messages = [
            ("system", system_prompt),
        ]

        # Add history
        for human, ai in history:
            messages.insert(-1, ("human", human))
            messages.insert(-1, ("assistant", ai))

        # Add current message
        messages.append(("human", message))

        # Call LLM
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"Error: {str(e)}"


# %% [markdown]
#
# ## Teil 5: Vollständiges Gradio Interface

# %%
# TODO: Create ChatInterface with additional inputs

# %% [markdown]
#
# ## Teil 6: Zusatzfunktionen (Optional)
#
# **Weitere Features, die Sie hinzufügen können**:
# - Export-Button für Konversation
# - Token-Zähler und Kosten-Tracker
# - Verschiedene Modelle zur Auswahl
# - Custom Styling/Theme
# - Beispiel-Prompts (Buttons)

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
# ## Beispiel-Anwendung: Kundenservice-Bot
#
# **Szenario**: Chatbot für einen Online-Shop
# - Persönlichkeit: Freundlich und hilfreich
# - Beantwortet Fragen zu Produkten
# - Hilft bei Bestellungen
# - Professionelles Interface

# %%
CUSTOMER_SERVICE_PROMPT = """\
Du bist ein freundlicher Kundenservice-Mitarbeiter für einen Online-Shop.
Du hilfst Kunden bei Fragen zu Produkten, Bestellungen und Versand.
Sei höflich, hilfsbereit und professionell."""

# %% [markdown]
#
# Erstellen des Chatbots

# %%

# %% [markdown]
#
# Erstellen des Interfaces

# %%

# %% [markdown]
#
# Starten des Chatbots

# %%
