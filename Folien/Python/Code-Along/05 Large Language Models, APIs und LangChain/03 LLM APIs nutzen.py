# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LLM APIs nutzen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Von der Chat-Oberfläche zum Code
#
# - Bisher: LLMs über Webseiten nutzen (ChatGPT, Claude, etc.)
# - Jetzt: LLMs **programmatisch** nutzen
# - **Warum?** Um eigene Anwendungen zu bauen!
# - **Wie?** Über APIs (Application Programming Interfaces)

# %% [markdown]
#
# ## Rückblick: REST APIs
#
# - Wir haben REST APIs bereits in Abschnitt 01 kennengelernt
# - **API**: Schnittstelle für Programm-zu-Programm-Kommunikation
# - **REST**: Standard für Web-APIs
# - **HTTP-Methoden**: GET, POST, PUT, DELETE
#
# LLM-APIs funktionieren genauso!

# %%
import requests

# %%
import json

# %%
import os

# %% [markdown]
#
# ## Hauptanbieter von LLM APIs
#
# - **OpenAI**: GPT-3.5, GPT-4
# - **Anthropic**: Claude (verschiedene Versionen)
# - **Google**: Gemini
# - **Meta**: Llama (lokal oder über Partner)
#
# Heute: OpenAI und Anthropic APIs

# %% [markdown]
#
# ## API-Schlüssel: Authentifizierung
#
# - **API-Schlüssel**: Wie ein Passwort für den API-Zugriff
# - Wo bekommen?
#   - OpenAI: platform.openai.com/api-keys
#   - Anthropic: console.anthropic.com
# - **WICHTIG**: API-Schlüssel NIEMALS im Code speichern!
# - Besser: Umgebungsvariablen nutzen

# %% [markdown]
#
# ## API-Schlüssel sicher speichern
#
# ```bash
# # In .env Datei (nicht in Git committen!)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# ```
#
# ```python
# # In Python-Code
# import os
# api_key = os.getenv("OPENAI_API_KEY")
# ```

# %% [markdown]
#
# ## Option 1: Reine HTTP-Requests
#
# - Nutzt die `requests` Bibliothek
# - Zeigt, was wirklich passiert
# - Gut zum Verständnis
# - Etwas umständlich für produktiven Code

# %% [markdown]
#
# ## Beispiel: OpenAI mit requests

# %%
# Example: Making a completion request (raw HTTP)
def call_openai_raw(prompt, api_key):
    """Call OpenAI API using raw HTTP requests"""
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

# %% [markdown]
#
# ## Was sehen wir hier?
#
# - **URL**: Endpoint der API
# - **Headers**: Authentifizierung (API-Schlüssel)
# - **Data**: Unsere Anfrage (Modell, Nachricht, Parameter)
# - **Response**: Antwort des LLMs

# %% [markdown]
#
# ## Option 2: OpenAI Python-Bibliothek
#
# - Einfacher zu nutzen
# - Bessere Fehlerbehandlung
# - Automatische Retries
# - **Installation**: `pip install openai`
# - Empfohlen für produktiven Code

# %% [markdown]
#
# ## Beispiel: OpenAI-Bibliothek

# %%
# !pip install openai

# %%
from openai import OpenAI

# %%
def call_openai_lib(prompt, api_key):
    """Call OpenAI using the official library"""
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# %% [markdown]
#
# ## Fehlerbehandlung
#
# - APIs können fehlschlagen
# - **Häufige Fehler**:
#   - Ungültiger API-Schlüssel (401)
#   - Rate Limit überschritten (429)
#   - Netzwerkprobleme (Timeout)
#   - Ungültige Anfrage (400)
# - **Wichtig**: Immer Fehler behandeln!

# %%
from openai import OpenAI, AuthenticationError, RateLimitError


# %%
def call_llm_safely(prompt, api_key):
    """Call LLM with error handling"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# %% [markdown]
#
# ## Konversation mit Verlauf
#
# - Für Chatbots: Gesprächsverlauf speichern
# - Liste von Nachrichten mit Rollen:
#   - `system`: Anweisungen an das LLM
#   - `user`: Benutzereingabe
#   - `assistant`: Antworten des LLMs
# - LLM sieht gesamten Verlauf

# %%
from openai import OpenAI

# %%
def chatbot_with_history():
    """Simple chatbot with conversation history"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    # TODO: Implement conversation loop

# %%
import os

# %%
# api_key = os.getenv("OPENAI_API_KEY")
# assert api_key is not None, "Please set the OPENAI_API_KEY environment variable."

# %%

# %% [markdown]
#
# ## Kosten und Rate Limits
#
# - **Kosten**: Pro Token (Eingabe + Ausgabe)
#   - GPT-3.5: ~$0.001 pro 1000 Tokens
#   - GPT-4: ~$0.03 pro 1000 Tokens
# - **Rate Limits**: Maximale Anfragen pro Minute
#   - Unterschiedlich je nach Tarif
#   - Bei Überschreitung: Warten oder Retry
#
# **Tipp**: Mit günstigen Modellen testen!

# %% [markdown]
#
# ## Zusammenfassung
#
# - LLM APIs ermöglichen programmatischen Zugriff
# - Zwei Optionen: Reine HTTP-Requests oder Bibliotheken
# - Wichtig: API-Schlüssel sicher speichern
# - Fehlerbehandlung ist essenziell
# - Kosten und Rate Limits beachten
#
# **Nächster Schritt**: Gradio für schöne Benutzeroberflächen!

# %% [markdown]
#
# ## Workshop: Erste API-Aufrufe
#
# **Aufgaben:**
# 1. API-Schlüssel einrichten (OpenAI oder Anthropic)
# 2. Ersten API-Aufruf machen
# 3. Einfachen Chatbot bauen
# 4. Fehlerbehandlung hinzufügen
# 5. Kosten tracken

# %%
