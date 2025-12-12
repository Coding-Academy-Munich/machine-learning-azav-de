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
# - Fehlerbehandlung selbst schreiben
#
# Besser: Eine fertige Bibliothek nutzen!
#
# - **Einfachere Syntax**
# - **Automatische Fehlerbehandlung** mit spezifischen Exceptions
# - **Token-Tracking** für Kostenüberwachung
# - **Automatische Retries** bei Netzwerk- und Serverfehlern

# %%
# !pip install openai

# %%
from openai import OpenAI
from dotenv import load_dotenv
import os

# %%
load_dotenv()

# %% [markdown]
#
# ## Die `OpenAI`-Klasse
#
# - Initialisierung mit API-Schlüssel
# - Optionale `base_url` für OpenRouter (anstatt OpenAI)
# - Methode `chat.completions.create()` für Textgenerierung

# %%
api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1"
model = "mistralai/ministral-14b-2512"

# %%

# %% [markdown]
#
# ### Die Chat Completions API
#
# - API für Textgenerierung mit Message-Format
# - Parameter: `model` und `messages` (Liste von Nachrichten)
# - Ergebnis: `response.choices[0].message.content`

# %%

# %%

# %%


# %% [markdown]
#
# ## Fehlerbehandlung mit der Bibliothek
#
# Die Bibliothek bietet **spezifische Exceptions**:
#
# - `AuthenticationError`: Ungültiger API-Schlüssel
# - `RateLimitError`: Rate Limit überschritten
# - `APIError`: Allgemeiner API-Fehler
#
# Viel besser als HTTP-Statuscodes prüfen!

# %%
from openai import AuthenticationError, RateLimitError, APIError


# %%
def ask_llm_safely(prompt, model=model):
    """Ask LLM with error handling."""
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


# %%
result = ask_llm_safely("What is a Large Language Model?")
print(result)

# %% [markdown]
#
# ## Token-Nutzung tracken
#
# **Kosten entstehen pro Token** (Eingabe + Ausgabe)
#
# Die Bibliothek macht Token-Tracking einfach:

# %%

# %%

# %%
print(f"Prompt tokens:     {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens:      {response.usage.total_tokens}")

# %% [markdown]
#
# ## Automatische Retries
#
# Die Bibliothek wiederholt bestimmte Anfragen **automatisch**:
#
# - **Standardmäßig 2 Wiederholungen** mit exponentiellem Backoff
# - Funktioniert bei: Verbindungsfehlern, Timeouts (408), Rate Limits (429),
#   Serverfehlern (500+)
#
# **Wichtig:** `AuthenticationError` (401) wird **nicht** wiederholt!

# %% [markdown]
#
# ### Retries konfigurieren
#
# Mit dem Parameter `max_retries`:

# %%
client_no_retry = OpenAI(api_key=api_key, base_url=url, max_retries=0)

# %%
client_many_retries = OpenAI(api_key=api_key, base_url=url, max_retries=5)

# %% [markdown]
#
# ### Retries pro Anfrage
#
# Mit `with_options()` kann man Retries für einzelne Anfragen ändern:

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# Die OpenAI-Bibliothek vereinfacht LLM-Aufrufe:
#
# - **Chat Completions API**: `client.chat.completions.create(model, messages)`
# - **Spezifische Exceptions** für verschiedene Fehlertypen
# - **Token-Tracking** mit `response.usage`
# - **Automatische Retries** bei Netzwerk-/Serverfehlern (konfigurierbar)
# - Konfigurierbar für OpenRouter mit `base_url`

# %% [markdown]
#
# ## Mini-Workshop: Client konfigurieren
#
# **Aufgabe**: Erstellen Sie zwei verschiedene OpenAI-Clients:
#
# 1. Einen Client für **schnelle, einfache Anfragen** (ohne Retries)
# 2. Einen Client für **wichtige Anfragen** (mit 5 Retries)
#
# Schreiben Sie dann eine Funktion `ask_important()`, die den zweiten Client
# nutzt und bei `AuthenticationError` eine hilfreiche Nachricht ausgibt.

# %% [markdown]
#
# ### Hinweise
#
# - Nutzen Sie den Parameter `max_retries` beim Erstellen des Clients
# - `AuthenticationError` wird **nicht** automatisch wiederholt
# - Geben Sie die Token-Nutzung mit `response.usage` aus

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Workshop (Optional): Chatbot mit der OpenAI-Bibliothek
#
# In der vorherigen Lektion haben wir einen Chatbot mit `requests` gebaut.
#
# **Aufgabe**: Bauen Sie den Chatbot mit der OpenAI-Bibliothek nach!
#
# Vergleichen Sie:
# - Wie viel einfacher ist der Code?
# - Welche Vorteile bietet die Bibliothek?

# %% [markdown]
#
# ### Hinweise
#
# - Nutzen Sie `client.chat.completions.create()` statt `requests.post()`
# - Die Message-Liste funktioniert genauso wie bei `requests`
# - Antwort: `response.choices[0].message.content`
# - **Bonus**: Tracken Sie die Token-Nutzung pro Nachricht

# %%

# %%

# %%

# %%

# %%
