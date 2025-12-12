# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Gradio: Chatbot Interface</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import gradio as gr

# %% [markdown]
#
# ## Chatbot-Interface
#
# - Gradio hat spezielles **Chatbot-Interface**
# - Zeigt Unterhaltungsverlauf
# - Perfekt für LLM-Anwendungen

# %% [markdown]
#
# ### Stub für eine Chatbot-Funktion
#
# ```python
# def respond(message, history):
#    ...
# ```
#
# - `message`: Neue Nachricht des Nutzers
# - `history`: Liste vorheriger Nachrichten
#    - Jede Nachricht ist ein Dictionary:<br>`{"role": "user"/"assistant", "content": ...}`

# %%

# %% [markdown]
#
# ### Chat Interface
#
# `gr.ChatInterface` erzeugt ein vollständiges Chat Interface:
#
# - `fn`: Die Chat-Funktion
# - `type`: Art der History (`"messages"` für OpenAI-Format)
# - `title` und `description`: Text für die Oberfläche

# %%

# %%
