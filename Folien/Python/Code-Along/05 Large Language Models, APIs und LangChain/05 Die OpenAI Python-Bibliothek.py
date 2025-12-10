# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Die OpenAI Python-Bibliothek</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Warum eine Bibliothek nutzen?
#
# Bisher: Rohe HTTP-Anfragen mit `requests`
#
# - Funktioniert, aber umständlich
# - Viel Boilerplate-Code
# - Fehlerbehandlung selbst schreiben
#
# Besser: Eine fertige Bibliothek nutzen!

# %% [markdown]
#
# ## Vorteile der OpenAI-Bibliothek
#
# - Einfachere Syntax
# - Automatische Fehlerbehandlung
# - Automatische Retries bei Fehlern
# - Typisierte Objekte statt Dictionaries
# - Gut dokumentiert und weit verbreitet

# %% [markdown]
#
# ## Installation und Import

# %%
# !pip install openai

# %%
from openai import OpenAI

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# %% [markdown]
#
# ## OpenAI-Client für OpenRouter konfigurieren
#
# Die OpenAI-Bibliothek kann mit **jedem** OpenAI-kompatiblen Dienst verwendet werden.
#
# Wir müssen nur:
# - Den API-Schlüssel setzen
# - Die **Base URL** auf OpenRouter ändern

# %%
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# %% [markdown]
#
# ## Unser erster Aufruf mit der Bibliothek

# %%

# %%

# %% [markdown]
#
# ## Das Response-Objekt
#
# Die Bibliothek gibt ein **Objekt** zurück, kein Dictionary:

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Vergleich: requests vs. Bibliothek
#
# **Mit requests:**
# ```python
# result["choices"][0]["message"]["content"]
# ```
#
# **Mit der Bibliothek:**
# ```python
# response.choices[0].message.content
# ```
#
# Punkt-Notation statt Dictionary-Zugriff!

# %% [markdown]
#
# ## Eine Hilfsfunktion

# %%
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# %%
def ask_llm(prompt, model="mistralai/ministral-14b-2512"):
    """Ask an LLM a question and return the response."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    # TODO: Return the response content
    pass

# %%

# %% [markdown]
#
# ## Fehlerbehandlung mit der Bibliothek
#
# Die Bibliothek bietet spezifische Exceptions:

# %%
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# %%
def ask_llm_safely(prompt, model="mistralai/ministral-14b-2512"):
    """Ask LLM with error handling."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# %%

# %% [markdown]
#
# ## Konversation mit Verlauf
#
# Für echte Chatbots: Den Gesprächsverlauf speichern
#
# - Liste von Nachrichten verwalten
# - Nach jeder Antwort die Nachricht hinzufügen
# - Das LLM sieht den gesamten Verlauf

# %% [markdown]
#
# ## Beispiel: Einfacher Chatbot

# %%
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# %%
def run_chatbot():
    """Run a simple chatbot with conversation history."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print("Chatbot gestartet! Geben Sie 'quit' zum Beenden ein.")
    print("Chatbot started! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # TODO: Add user message, get response, add assistant message
        pass

# %% [markdown]
#
# ## Den Chatbot starten
#
# Entfernen Sie das Kommentarzeichen, um den Chatbot zu starten:

# %%

# %% [markdown]
#
# ## Konversation ohne interaktive Schleife
#
# Wir können auch einzelne Konversationsschritte zeigen:

# %%
messages = [
    {"role": "system", "content": "You are a helpful assistant. Be brief."}
]

# %%

# %% [markdown]
#
# ## Das Modell erinnert sich

# %%

# %% [markdown]
#
# ## Der gesamte Verlauf

# %%

# %% [markdown]
#
# ## Kosten und Rate Limits
#
# **Kosten:**
# - Pro Token (Eingabe + Ausgabe)
# - Unterschiedlich je nach Modell
# - GPT-4: teurer, aber besser
# - GPT-3.5 / Llama: günstiger
#
# **Tipp:** Mit günstigen Modellen entwickeln und testen!

# %% [markdown]
#
# ## Rate Limits
#
# - Maximale Anfragen pro Minute/Tag
# - Unterschiedlich je nach Anbieter und Tarif
# - Bei Überschreitung: `RateLimitError`
#
# **Tipps:**
# - Pausen zwischen Anfragen einbauen
# - Fehler abfangen und warten
# - Die Bibliothek hat automatische Retries!

# %% [markdown]
#
# ## Token-Nutzung anzeigen

# %%

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - Die OpenAI-Bibliothek vereinfacht LLM-Aufrufe
# - Konfigurierbar für OpenRouter mit `base_url`
# - Punkt-Notation statt Dictionary-Zugriff
# - Eingebaute Fehlerbehandlung
# - Konversationsverlauf durch Message-Liste

# %% [markdown]
#
# ## Workshop: LLM-Anwendungen bauen
#
# **Aufgabe 1:** OpenAI-Client einrichten
#
# Erstellen Sie einen OpenAI-Client, der OpenRouter nutzt.

# %%

# %% [markdown]
#
# **Aufgabe 2:** Einfache Completion
#
# Machen Sie eine einfache Anfrage an das LLM.

# %%

# %% [markdown]
#
# **Aufgabe 3:** Verschiedene Modelle
#
# Probieren Sie verschiedene Modelle aus und vergleichen Sie die Antworten.

# %%
models = [
    "mistralai/ministral-14b-2512",
    "openai/gpt-4o-mini",
]

# %%
prompt = "What is machine learning? One sentence."

# %%

# %% [markdown]
#
# **Aufgabe 4:** Konversation bauen
#
# Erstellen Sie eine kurze Konversation mit Verlauf.

# %%

# %%

# %%

# %% [markdown]
#
# **Aufgabe 5:** Temperature experimentieren
#
# Probieren Sie verschiedene Temperature-Werte (0.0 bis 1.0).
# Niedrige Werte = deterministischer, hohe Werte = kreativer.

# %%
prompt = "Write a one-sentence story about a robot."

# %%

# %%
