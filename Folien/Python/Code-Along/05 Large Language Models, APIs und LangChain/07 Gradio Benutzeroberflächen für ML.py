# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Gradio: Benutzeroberflächen für ML</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Warum Benutzeroberflächen?
#
# - Kommandozeilen-Apps sind funktional, aber...
# - **Nicht benutzerfreundlich** für Nicht-Programmierer
# - **Nicht beeindruckend** im Portfolio
# - **Nicht teilbar** ohne Code
#
# **Lösung**: Einfache Web-Interfaces!

# %% [markdown]
#
# ## Was ist Gradio?
#
# - **Python-Bibliothek** für ML-Interfaces
# - **Wenige Zeilen Code** → Professionelle Web-UI
# - **Funktioniert mit jedem Python-Code**
# - **Teilbar** via Link
# - **Kostenlos und Open Source**
#
# **Ideal für**: Demos, Prototypen, Portfolio-Projekte

# %% [markdown]
#
# ## Installation
#
# ```bash
# pip install gradio
# ```
#
# **Das war's!** Keine weitere Konfiguration nötig.

# %%
# !pip install gradio --root-user-action=ignore

# %%
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os

# %%
load_dotenv()

# %% [markdown]
#
# ## Erstes Beispiel: Hello World

# %%

# %%

# %%

# %% [markdown]
#
# ## Was passiert hier?
#
# 1. `fn=greet`: Unsere Python-Funktion
# 2. `inputs="text"`: Textfeld für Eingabe
# 3. `outputs="text"`: Textfeld für Ausgabe
# 4. `demo.launch()`: Startet Web-Server
#
# Gradio erstellt automatisch die Web-Oberfläche!

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
