# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Ein einfacher Chatbot mit REST API</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bisher können
#
# - REST-APIs aufrufen mit `requests`
# - LLM-Anfragen senden und Antworten extrahieren
# - Fehler abfangen und behandeln
#
# **Jetzt**: Einen echten Chatbot bauen!
#
# Das Ziel: Zeigen, dass wir mit dem Gelernten bereits
# etwas Beeindruckendes bauen können.

# %% [markdown]
#
# ## Das Problem: Einzelne Anfragen
#
# Unsere `call_llm_safely` Funktion sendet **eine** Nachricht und bekommt **eine** Antwort.
#
# ```python
# call_llm_safely("Hallo, ich bin Max")
# # "Hallo Max! Wie kann ich helfen?"
#
# call_llm_safely("Wie heiße ich?")
# # "Das weiß ich leider nicht..."  # Das Modell hat vergessen!
# ```
#
# **Problem**: Das Modell "vergisst" alles nach jeder Anfrage.

# %% [markdown]
#
# ## Die Lösung: Konversationsverlauf
#
# Erinnern Sie sich an die **Message-Liste**?
#
# ```python
# messages = [
#     {"role": "user", "content": "Hallo, ich bin Max"},
#     {"role": "assistant", "content": "Hallo Max!"},
#     {"role": "user", "content": "Wie heiße ich?"}
# ]
# ```
#
# Wenn wir **alle** Nachrichten mitsenden, kann das Modell sich "erinnern"!

# %% [markdown]
#
# ## Von der Funktion zur Klasse
#
# Wir brauchen eine Klasse, die:
#
# 1. Die **Konfiguration** speichert (URL, API-Key, Modell)
# 2. Den **Konversationsverlauf** verwaltet
# 3. Neue Nachrichten **hinzufügt** und Antworten speichert
#
# So wird aus einzelnen Aufrufen eine echte Konversation!

# %%
import requests
from dotenv import load_dotenv
import os

load_dotenv()


# %% [markdown]
#
# ## Die Chatbot-Klasse: Grundgerüst

# %%
class SimpleChatbot:
    """A simple chatbot using the REST API."""

    def __init__(self, model="mistralai/ministral-14b-2512"):
        """Initialize the chatbot with configuration."""
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        self.model = model

    def chat(self, user_message):
        """Send a message and get a response."""
        ...


# %% [markdown]
#
# ## Die chat() Methode
#
# Diese Methode macht die eigentliche Arbeit:
#
# 1. User-Nachricht zur History hinzufügen
# 2. API-Anfrage mit **gesamter** History senden
# 3. Antwort extrahieren und zur History hinzufügen
# 4. Antwort zurückgeben

# %%
class SimpleChatbot:
    """A simple chatbot using the REST API."""

    def __init__(self, model="mistralai/ministral-14b-2512"):
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        self.model = model

    def chat(self, user_message):
        """Send a message and get a response."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_message}],
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()

            assistant_message = result["choices"][0]["message"]["content"]
            return assistant_message

        except requests.HTTPError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"


# %% [markdown]
#
# ## Den Chatbot testen

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Es funktioniert!
#
# Der Chatbot **erinnert** sich an den Kontext:
#
# - Er weiß, dass wir Max heißen
# - Er weiß, dass wir Python lernen
# - Er kann auf frühere Fragen Bezug nehmen
#
# Alles mit dem, was wir bereits gelernt haben!

# %% [markdown]
#
# ## Den Verlauf anschauen
#
# Wir können sehen, was der Chatbot "weiß":

# %%
for msg in bot.messages:
    role = msg["role"].upper()
    content = (
        msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
    )
    print(f"{role}: {content}")


# %% [markdown]
#
# ## Interaktive Chat-Schleife
#
# Jetzt machen wir den Chatbot **interaktiv**:

# %%
def run_chatbot():
    """Run an interactive chatbot session."""
    bot = SimpleChatbot()

    print("=" * 50)
    print("Chatbot gestartet!")
    print("Geben Sie 'quit' ein zum Beenden.")
    print("=" * 50)

    while True:
        user_input = input("\nSie: ")

        if user_input.lower() == "quit":
            print("\nAuf Wiedersehen!")
            break

        response = bot.chat(user_input)
        print(f"\nBot: {response}")


# %%
def run_chatbot():
    """Run an interactive chatbot session."""
    bot = SimpleChatbot()

    print("=" * 50)
    print("Chatbot started!")
    print("Type 'quit' to exit.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "quit":
            print("\nGoodbye!")
            break

        response = bot.chat(user_input)
        print(f"\nBot: {response}")


# %% [markdown]
#
# ## Den interaktiven Chatbot starten
#
# Entfernen Sie das Kommentarzeichen, um den Chatbot zu starten:

# %%
# # run_chatbot()

# %% [markdown]
#
# ## Erweiterung: System-Prompt
#
# Mit einem **System-Prompt** können wir dem Chatbot eine Persönlichkeit geben:

# %%
class ChatbotWithPersonality:
    """A chatbot with a customizable personality."""

    def __init__(self, system_prompt, model="mistralai/ministral-14b-2512"):
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        self.model = model
        # Start with system message
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, user_message):
        """Send a message and get a response."""
        self.messages.append({"role": "user", "content": user_message})

        data = {"model": self.model, "messages": self.messages}

        try:
            response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()

            assistant_message = result["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": assistant_message})

            return assistant_message

        except Exception as e:
            return f"Error: {e}"

# %% [markdown]
#
# ## Beispiel: Ein hilfreicher Python-Tutor

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Was haben wir erreicht?
#
# Mit nur **~30 Zeilen Code** haben wir:
#
# - Einen funktionierenden Chatbot gebaut
# - Der sich an Konversationen erinnert
# - Der eine anpassbare Persönlichkeit hat
# - Der Fehler abfängt
#
# **Alles mit dem, was wir in dieser Lektion gelernt haben!**

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Konversationsverlauf**: Alle Nachrichten in einer Liste speichern
# - **Klassen**: Konfiguration und State zusammen verwalten
# - **Chat-Schleife**: Mit `input()` interaktiv machen
# - **System-Prompt**: Persönlichkeit definieren
#
# **Nächster Schritt**: Die OpenAI-Bibliothek macht das noch einfacher!

# %% [markdown]
#
# ## Mini-Workshop: Chatbot erweitern
#
# **Aufgaben:**
#
# 1. Erstellen Sie einen Chatbot mit einem eigenen System-Prompt (z.B. ein Koch,
#    ein Reiseführer, ein Geschichtenerzähler)
# 2. Fügen Sie eine Methode `clear_history()` hinzu, die den Verlauf löscht
# 3. **Bonus**: Fügen Sie eine Methode `save_conversation()` hinzu, die den
#    Verlauf in eine Textdatei speichert

# %%

# %%

# %%

# %%
