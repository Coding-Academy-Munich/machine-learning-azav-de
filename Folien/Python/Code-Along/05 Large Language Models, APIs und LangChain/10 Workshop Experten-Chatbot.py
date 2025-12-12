# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: Experten-Chatbot</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os

# %%
load_dotenv()

# %%
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# %% [markdown]
#
# ## Workshop: Experten-Chatbot
#
# **Ziel**: Einen spezialisierten Chatbot mit eigenem System Prompt bauen!
#
# Der Chatbot soll:
# - Ein Experte auf einem Gebiet sein (z.B. Koch, Reiseberater, Fitness-Trainer)
# - Nur zu seinem Thema antworten
# - Eine eigene "Persönlichkeit" haben

# %% [markdown]
#
# ### Aufgabe 1: System Prompt erstellen
#
# Erstellen Sie einen System Prompt für Ihren Experten-Chatbot.
#
# **Beispiel** (Koch):
# ```
# Du bist ein erfahrener Koch mit 20 Jahren Erfahrung.
# Du gibst nur Ratschläge zu Kochen und Rezepten.
# Wenn jemand nach anderen Themen fragt, lenkst du
# freundlich zurück zum Thema Kochen.
# ```

# %%
EXPERT_PROMPT = """
# TODO: Write your expert system prompt here
"""


# %% [markdown]
#
# ### Aufgabe 2: Chatbot-Funktion erstellen
#
# Erstellen Sie die Funktion für Ihren Experten-Chatbot.

# %%
def expert_respond(message, history):
    """Expert chatbot response function"""
    # TODO: Use EXPERT_PROMPT and call the LLM
    pass

# %% [markdown]
#
# ### Aufgabe 3: Gradio Interface erstellen
#
# Erstellen Sie das ChatInterface mit passendem Titel und Beschreibung.

# %%
# TODO: Create gr.ChatInterface for your expert chatbot
# expert_chatbot = gr.ChatInterface(...)

# %%

# %% [markdown]
#
# ### Bonus-Aufgaben
#
# 1. **Testen**: Versuchen Sie, den Chatbot mit Off-Topic-Fragen zu verwirren
# 2. **Verbessern**: Passen Sie den System Prompt an, falls der Chatbot zu leicht abgelenkt wird
# 3. **Zweiter Experte**: Erstellen Sie einen zweiten Chatbot mit anderem Thema
