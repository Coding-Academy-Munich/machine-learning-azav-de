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
# !pip install gradio

# %%
import gradio as gr

# %% [markdown]
#
# ## Erstes Beispiel: Hello World

# %%
def greet(name):
    """Simple greeting function"""
    # TODO: Return greeting

# %%
# Create Gradio interface
# TODO: Create interface with gr.Interface()

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

# %%
def respond(message, history):
    """Simple chatbot response function"""
    # history is list of [user_msg, bot_msg] pairs
    # TODO: Implement response logic
    return "Response goes here"

# %%
# Create chatbot interface
# TODO: Create with gr.ChatInterface()

# %%

# %% [markdown]
#
# ## Eingabe-Typen
#
# Gradio unterstützt viele Input-Typen:
# - `"text"`: Textfeld
# - `gr.Slider()`: Schieberegler
# - `gr.Dropdown()`: Auswahlmenü
# - `gr.Checkbox()`: Kontrollkästchen
# - `gr.Image()`: Bild-Upload
# - `gr.File()`: Datei-Upload
# - Und viele mehr!

# %% [markdown]
#
# ## Beispiel: Chatbot mit Einstellungen

# %%
def enhanced_chatbot(message, history, personality, temperature):
    """Chatbot with personality and temperature settings"""
    # TODO: Implement with settings
    pass

# %%
# Create interface with additional inputs
# TODO: Create with gr.ChatInterface and additional_inputs

# %% [markdown]
#
# ## Styling und Themes
#
# - Gradio kommt mit verschiedenen Themes
# - Anpassbar mit CSS
# - Einfache Parameter für Layout

# %%
# Example with custom theme
demo_styled = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(placeholder="Geben Sie Ihren Namen ein / Enter your name"),
    outputs=gr.Textbox(label="Begrüßung / Greeting"),
    title="Stylish Greeter",
    theme=gr.themes.Soft()
)

# Launch
# demo_styled.launch()

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
# ## Workshop-Aufgabe
#
# **Ziel**: Chatbot aus vorheriger Lektion mit Gradio-UI ausstatten
#
# **Schritte**:
# 1. Chatbot-Funktion aus topic_070 nehmen
# 2. Gradio ChatInterface erstellen
# 3. Persönlichkeits-Dropdown hinzufügen
# 4. Temperatur-Slider hinzufügen
# 5. Testen und anpassen

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Gradio**: Schnell professionelle UIs erstellen
# - **Wenige Zeilen Code**: Große Wirkung
# - **Viele Input/Output-Typen**: Flexibel
# - **Teilbar**: Lokal oder öffentlich
# - **Perfekt für ML-Demos**: Portfolio-Qualität
#
# **Nächster Schritt**: LangChain für noch leistungsfähigere Anwendungen!

# %%
