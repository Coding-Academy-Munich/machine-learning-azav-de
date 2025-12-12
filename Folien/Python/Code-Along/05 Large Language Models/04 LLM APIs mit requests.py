# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LLM APIs mit requests</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## LLM API Anbieter
#
# Es gibt viele Anbieter von LLM-APIs:
#
# | Anbieter       | Modelle                     | Besonderheit                             |
# | ----------     | ---------                   | --------------                           |
# | OpenAI         | GPT-5.1                     | Marktführer                              |
# | Anthropic      | Claude 4.5                  | Alignment, Coding                        |
# | Google         | Gemini 3                    | Multimodal                               |
# | Meta           | Llama 2,3,4                 | Open Source                              |
# | Mistral        | Ministral, Mixtral, Medium  | Europäisch, teilweise OS                 |
# | DeepSeek       | V3, R1, V3.2, V3.2 Speciale | SotA Open Source                         |
# | MoonshotAI     | Kimi K2                     | SotA Open Source                         |
# | Z.AI           | GLM 4.6                     | SotA Open Source                         |
# | Qwen (Alibaba) | Qwen 3                      | Sehr gute "kleinere" Open Source Modelle |

# %% [markdown]
#
# ## OpenRouter: Ein Gateway zu allen Modellen
#
# - **OpenRouter** bietet Zugang zu allen großen Anbietern
# - Ein API-Schlüssel für alles
# - OpenAI-kompatible API
# - Einfacher Wechsel zwischen Modellen
#
# Wir nutzen OpenRouter für unsere Beispiele!

# %% [markdown]
#
# ## Wie funktioniert ein Chat mit einem LLM?
#
# In der Chat-Oberfläche (z.B. ChatGPT):
#
# ```
# Sie: Was ist Python?
# Assistent: Python ist eine Programmiersprache...
# Sie: Wofür wird es verwendet?
# Assistent: Python wird für Webentwicklung, Data Science...
# ```
#
# Jede Nachricht hat einen **Absender** und einen **Inhalt**.

# %% [markdown]
#
# ## Von der Oberfläche zur API
#
# Die API braucht dieselben Informationen:
#
# - **Wer** hat die Nachricht geschrieben? → `role`
# - **Was** steht in der Nachricht? → `content`
#
# Jede Nachricht wird ein **Dictionary**:
#
# ```python
# {"role": "user", "content": "Was ist Python?"}
# ```

# %% [markdown]
#
# ## Warum eine Liste von Dictionaries?
#
# Das LLM braucht den **gesamten Gesprächsverlauf**:
#
# ```python
# messages = [
#     {"role": "user", "content": "Was ist Python?"},
#     {"role": "assistant", "content": "Python ist..."},
#     {"role": "user", "content": "Wofür wird es verwendet?"}
# ]
# ```
#
# - Jedes Dictionary = eine Nachricht
# - Die Liste = der gesamte Chat-Verlauf
# - `role` = wer hat geschrieben (user/assistant)

# %% [markdown]
#
# ## Die drei Rollen
#
# - **`user`**: Ihre Nachrichten (Fragen, Anweisungen)
# - **`assistant`**: Antworten des LLMs
# - **`system`**: Anweisungen an das LLM (optional)
#
# ```python
# messages = [
#     {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
#     {"role": "user", "content": "Hallo!"}
# ]
# ```
#
# Mehr zu Rollen in einem späteren Kapitel!

# %% [markdown]
#
# ## Jetzt: Unsere erste Anfrage an ein LLM
#
# Wir nutzen:
# - Die `requests` Bibliothek
# - OpenRouter als Anbieter
# - Das Modell `mistralai/ministral-14b-2512`

# %%
import requests
from dotenv import load_dotenv
import os

# %%
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# %% [markdown]
#
# ## Die Anfrage vorbereiten

# %%
url = "https://openrouter.ai/api/v1/chat/completions"

# %%
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# %%
data = {
    "model": "mistralai/ministral-14b-2512",
    "messages": [
        {"role": "user", "content": "What is Python? Answer in one sentence."}
    ],
}


# %% [markdown]
#
# ## Die Anfrage senden

# %%

# %%

# %% [markdown]
#
# ## Die Antwort untersuchen

# %%

# %% [markdown]
#
# ## Die Antwortstruktur verstehen
#
# Die Antwort ist ein verschachteltes Dictionary:
#
# ```python
# {
#     "id": "gen-...",
#     "model": "mistralai/ministral-14b-2512",
#     "choices": [
#         {
#             "message": {
#                 "role": "assistant",
#                 "content": "Python ist..."
#             }
#         }
#     ],
#     "usage": {"prompt_tokens": 10, "completion_tokens": 20}
# }
# ```

# %% [markdown]
#
# ## Warum `choices[0]["message"]["content"]`?
#
# - **`choices`**: Eine Liste, weil das Modell mehrere Antworten geben könnte
# - **`[0]`**: Wir wollen die erste (und meist einzige) Antwort
# - **`["message"]`**: Das Nachrichtenobjekt (wie unsere Eingabe)
# - **`["content"]`**: Der eigentliche Text der Antwort

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Die Antwort extrahieren

# %%

# %%

# %% [markdown]
#
# ## Eine Funktion für LLM-Aufrufe
#
# Fassen wir das in einer wiederverwendbaren Funktion zusammen:

# %%
def call_llm(prompt, model="mistralai/ministral-14b-2512"):
    """Call an LLM via OpenRouter and return the response text."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}

    # TODO: Send request and return response
    pass


# %% [markdown]
#
# ## Die Funktion testen

# %%

# %%

# %% [markdown]
#
# ## Verschiedene Modelle testen

# %%
models = [
    "mistralai/ministral-14b-2512",
    "openai/gpt-5-mini",
    "anthropic/claude-haiku-4.5",
]

# %%
prompt = "Write a Python function to print the numbers from 1 to 100."

# %%

# %% [markdown]
#
# ## Fehlerbehandlung
#
# APIs können aus verschiedenen Gründen fehlschlagen:
#
# - **401 Unauthorized**: Ungültiger API-Schlüssel
# - **429 Too Many Requests**: Rate Limit überschritten
# - **400 Bad Request**: Fehlerhafte Anfrage
# - **500 Internal Server Error**: Problem beim Anbieter

# %% [markdown]
#
# ## Beispiel: Ungültiger API-Schlüssel

# %%
bad_headers = {
    "Authorization": "Bearer invalid-key",
    "Content-Type": "application/json",
}


# %%

# %% [markdown]
#
# ## LLM-Funktion mit Fehlerbehandlung

# %%
def call_llm_safely(prompt, model="mistralai/ministral-14b-2512"):
    """Call LLM with error handling."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}

    # TODO: Add try/except for error handling
    pass


# %% [markdown]
#
# ## Die Funktion testen

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - LLM-APIs erwarten Nachrichten als **Liste von Dictionaries**
# - Jede Nachricht hat `role` und `content`
# - Die Antwort ist verschachtelt: `choices[0]["message"]["content"]`
# - Fehlerbehandlung ist wichtig!
# - OpenRouter ermöglicht einfachen Zugang zu vielen Modellen

# %% [markdown]
#
# ## Mini-Workshop: LLM-Aufrufe mit requests
#
# **Aufgaben:**
#
# 1. Schreiben Sie eine Anfrage an ein LLM, die nach der Hauptstadt von Deutschland fragt
# 2. Extrahieren Sie die Antwort aus dem Response-Dictionary
# 3. Probieren Sie ein anderes Modell aus (z.B. `openai/gpt-5-mini`)
# 4. Fügen Sie Fehlerbehandlung hinzu

# %%
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# %% [markdown]
#
# ### Aufgabe 1 & 2: Anfrage und Antwort extrahieren

# %%

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 3: Ein anderes Modell probieren

# %%

# %% [markdown]
#
# ### Aufgabe 4: Mit Fehlerbehandlung

# %%

# %%
