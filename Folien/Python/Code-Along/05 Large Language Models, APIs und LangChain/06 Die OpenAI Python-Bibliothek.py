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
# - **Automatische Retries** bei Fehlern

# %%
# !pip install openai

# %%
from openai import OpenAI, api_key
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
# - Methoden für verschiedene Endpunkte, z.B. `chat.completions.create()`

# %%
api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1"
model = "mistralai/ministral-14b-2512"

# %%
from openai import OpenAI

# %%

# %% [markdown]
#
# ### Chat mit dem LLM

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
# ## Rate Limits
#
# - Maximale Anfragen pro Minute/Tag
# - Bei Überschreitung: `RateLimitError`
#
# **Tipps:**
# - Pausen zwischen Anfragen einbauen
# - Fehler abfangen und warten
# - Die Bibliothek hat automatische Retries!

# %% [markdown]
#
# ## Zusammenfassung
#
# Die OpenAI-Bibliothek vereinfacht LLM-Aufrufe:
#
# - **Spezifische Exceptions** für verschiedene Fehlertypen
# - **Token-Tracking** mit `response.usage`
# - **Automatische Retries** bei Rate Limits
# - Konfigurierbar für OpenRouter mit `base_url`
#
# **Nächster Schritt**: LangChain für noch weniger Code!

# %% [markdown]
#
# ## Mini-Workshop: Fehlerbehandlung
#
# **Aufgabe**: Schreiben Sie eine Funktion, die:
# 1. Eine Anfrage an das LLM sendet
# 2. Bei `AuthenticationError` eine hilfreiche Nachricht ausgibt
# 3. Bei `RateLimitError` 5 Sekunden wartet und es nochmal versucht
# 4. Die Token-Nutzung am Ende ausgibt
#
# **Hinweise**:
# - Nutzen Sie `time.sleep(5)`, um 5 Sekunden zu warten
# - Sie können oft einen `RateLimitError` provozieren, wenn Sie eines der
#   kostenlosen Modelle nutzen, z.B. `allenai/olmo-3-32b-think:free`

# %%
import time

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
# - Die Message-Liste funktioniert genauso
# - Antwort: `response.choices[0].message.content`
# - **Bonus**: Tracken Sie die Token-Nutzung pro Nachricht

# %%

# %%

# %%

# %%

# %%

# %%
