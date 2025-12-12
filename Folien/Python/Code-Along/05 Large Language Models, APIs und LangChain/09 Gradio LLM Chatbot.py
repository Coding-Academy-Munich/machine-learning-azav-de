# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Gradio: LLM Chatbot</b>
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

# %% [markdown]
#
# ## LLM-Chatbot mit System Prompt
#
# Jetzt verbinden wir Gradio mit einem echten LLM:
#
# - Nutzen die OpenAI-Bibliothek
# - Fügen einen **System Prompt** hinzu
# - Der System Prompt definiert die "Persönlichkeit" des Chatbots

# %%
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# %%

# %%
def llm_respond(message, history):
    """LLM chatbot with system prompt"""
    # TODO: Build messages with system prompt and call LLM
    pass

# %%

# %%

# %% [markdown]
#
# ## Teilen Ihrer Anwendung
#
# - **Lokal**: `demo.launch()` - Nur auf Ihrem Computer
# - **Öffentlich**: `demo.launch(share=True)` - Temporärer öffentlicher Link
# - **HuggingFace Spaces**: Permanent hosten (kostenlos!)
#
# **Perfekt für Portfolio-Projekte!**

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Gradio**: Schnell professionelle UIs erstellen
# - **`gr.Interface`**: Einfache Eingabe/Ausgabe-Oberflächen
# - **`gr.ChatInterface`**: Chatbot mit History
# - **System Prompt**: Definiert Chatbot-Verhalten
# - **Teilbar**: Lokal oder öffentlich
#
# **Nächster Schritt**: RAG - Chatbots mit eigenem Wissen!
