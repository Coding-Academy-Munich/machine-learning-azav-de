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
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

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
    # history is list of {"role": ..., "content": ...} dicts
    # TODO: Implement response logic
    return "Response goes here"

# %%
# Create chatbot interface
# TODO: Create with gr.ChatInterface()

# %%

# %% [markdown]
#
# ## Ein echter LLM-Chatbot!
#
# Jetzt verbinden wir Gradio mit unserem LLM:
#
# - Nutzen die OpenAI-Bibliothek aus der vorherigen Lektion
# - Gradio übergibt die **gesamte History** an unsere Funktion
# - Wir müssen sie nur ins richtige Format bringen

# %%
# Create OpenAI client for OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# %%
def llm_respond(message, history):
    """Real LLM chatbot response function"""
    # TODO: Convert history and call LLM
    pass

# %%
# Create real LLM chatbot
llm_chatbot = gr.ChatInterface(
    fn=llm_respond,
    type="messages",
    title="LLM Chatbot",
    description="Ein echter Chatbot mit LLM! / A real chatbot with LLM!"
)

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
PERSONALITY_PROMPTS = {
    "Freundlich": "Du bist ein freundlicher, hilfsbereiter Assistent. Sei warmherzig und ermutigend.",
    "Professionell": "Du bist ein professioneller Assistent. Antworte sachlich und präzise.",
    "Lustig": "Du bist ein lustiger Assistent. Mache Witze und sei unterhaltsam, aber bleibe hilfreich."
}

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
# ## Workshop: Bild-Klassifikator mit Gradio
#
# **Ziel**: Eine Bildklassifikations-App mit professioneller Oberfläche bauen!
#
# Wir lernen:
# - `gr.Image()` für Bild-Upload
# - `gr.Label()` für Klassifikations-Ergebnisse
# - Mehrere Outputs kombinieren
# - Styling mit Themes

# %% [markdown]
#
# ### Schritt 1: Einfache Klassifikator-Funktion
#
# Zuerst erstellen wir eine Mock-Funktion, die Bilder "klassifiziert":

# %%
def classify_image(image):
    """Simple image classifier function"""
    # TODO: Return classification results as dictionary
    # Format: {"label": confidence, ...}
    pass

# %% [markdown]
#
# ### Schritt 2: Gradio Interface erstellen
#
# Jetzt verbinden wir die Funktion mit Gradio:

# %%
# Create classifier interface
# TODO: Create with gr.Interface, gr.Image input, gr.Label output

# %%

# %% [markdown]
#
# ### Schritt 3: Mehrere Outputs hinzufügen
#
# Wir können mehrere Informationen zurückgeben:

# %%
def classify_with_details(image):
    """Classifier with detailed output"""
    if image is None:
        return {"No image": 1.0}, "Kein Bild / No image", 0

    # Simulated predictions
    predictions = {
        "Katze / Cat": 0.75,
        "Hund / Dog": 0.15,
        "Vogel / Bird": 0.07,
        "Andere / Other": 0.03
    }

    # Get top prediction
    top_label = max(predictions, key=predictions.get)
    confidence = predictions[top_label]

    # Create summary text
    summary = f"Top: {top_label} ({confidence:.0%})"

    return predictions, summary, confidence

# %%
# Create interface with multiple outputs
# TODO: Create with list of outputs

# %%

# %% [markdown]
#
# ### Schritt 4: Mit Theme stylen
#
# Machen wir die App professioneller:

# %%
# Styled classifier with theme
styled_classifier = gr.Interface(
    fn=classify_with_details,
    inputs=gr.Image(
        type="pil",
        label="Bild hochladen / Upload Image",
        sources=["upload", "clipboard"]
    ),
    outputs=[
        gr.Label(num_top_classes=4, label="Klassifikation / Classification"),
        gr.Textbox(label="Ergebnis / Result"),
        gr.Number(label="Konfidenz / Confidence", precision=2)
    ],
    title="🖼️ ML Bild-Klassifikator / Image Classifier",
    description="Laden Sie ein Bild hoch und sehen Sie die KI-Vorhersage! / Upload an image and see the AI prediction!",
    theme=gr.themes.Soft(),
    examples=[
        # Add example images if available
        # ["path/to/example1.jpg"],
        # ["path/to/example2.jpg"],
    ]
)

# %%

# %% [markdown]
#
# ### Bonus-Aufgaben
#
# 1. **Echtes Modell**: Verwenden Sie ein vortrainiertes Modell (z.B. `torchvision.models`)
# 2. **Mehrere Bilder**: Erlauben Sie den Upload mehrerer Bilder
# 3. **Verlauf**: Zeigen Sie die letzten Klassifikationen an
# 4. **Export**: Fügen Sie einen Button zum Speichern der Ergebnisse hinzu

# %%
# Bonus: Example with error handling

def robust_classifier(image):
    """Classifier with error handling"""
    try:
        if image is None:
            return {"Fehler / Error": 1.0}, "Bitte Bild hochladen / Please upload image"

        # Simulated classification
        predictions = {
            "Katze / Cat": 0.75,
            "Hund / Dog": 0.15,
            "Vogel / Bird": 0.07,
            "Andere / Other": 0.03
        }

        top_label = max(predictions, key=predictions.get)
        return predictions, f"Erkannt / Detected: {top_label}"

    except Exception as e:
        return {"Fehler / Error": 1.0}, f"Fehler: {e} / Error: {e}"

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Gradio**: Schnell professionelle UIs erstellen
# - **Wenige Zeilen Code**: Große Wirkung
# - **Viele Input/Output-Typen**: Bilder, Text, Labels, Slider, ...
# - **Teilbar**: Lokal oder öffentlich
# - **Perfekt für ML-Demos**: Portfolio-Qualität
#
# **Nächster Schritt**: LangChain für leistungsfähige LLM-Anwendungen mit Chatbot-Workshop!

# %%
