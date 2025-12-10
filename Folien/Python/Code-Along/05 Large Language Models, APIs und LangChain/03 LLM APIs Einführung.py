# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LLM APIs: Einführung</b>
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
# ## Was können wir damit bauen?
#
# - **Chatbots:** Kundenservice, Assistenten
# - **Textverarbeitung:** Zusammenfassungen, Übersetzungen
# - **Code-Assistenten:** Automatische Code-Generierung
# - **Datenanalyse:** Automatische Berichte aus Daten
# - **Kreative Tools:** Geschichten, Marketing-Texte

# %% [markdown]
#
# ## Kurze Wiederholung: REST APIs
#
# Wir haben REST APIs bereits in Abschnitt 01 kennengelernt:
#
# - **API**: Schnittstelle für Programm-zu-Programm-Kommunikation
# - **REST**: Standard für Web-APIs
# - **HTTP-Methoden**: GET (abrufen), POST (senden)
# - **JSON**: Datenformat für Anfragen und Antworten
#
# LLM-APIs funktionieren genauso!

# %% [markdown]
#
# ## Das Muster für LLM-APIs
#
# ```python
# import requests
#
# headers = {
#     "Authorization": "Bearer MEIN_API_KEY",
#     "Content-Type": "application/json"
# }
#
# data = {"model": "...", "messages": [...]}
#
# response = requests.post(url, headers=headers, json=data)
# result = response.json()
# ```
#
# **Wichtig:** Wir brauchen einen API-Schlüssel!

# %% [markdown]
#
# ## API-Schlüssel: Was ist das?
#
# - **API-Schlüssel**: Wie ein Passwort für den API-Zugriff
# - Identifiziert Sie gegenüber dem Dienst
# - Wird für die Abrechnung verwendet
# - Jeder Anbieter hat eigene Schlüssel:
#   - OpenAI: `sk-...`
#   - Anthropic: `sk-ant-...`
#   - OpenRouter: `sk-or-...`

# %% [markdown]
#
# ## Wichtig: API-Schlüssel sicher aufbewahren!
#
# **NIEMALS** API-Schlüssel im Code speichern:
#
# ```python
# # FALSCH - Niemals so machen!
# api_key = "sk-abc123..."
# ```
#
# Warum nicht?
# - Code wird oft in Git gespeichert
# - Andere könnten den Schlüssel sehen
# - Sie müssten für deren Nutzung bezahlen!

# %% [markdown]
#
# ## Lösung: Umgebungsvariablen
#
# - Speichern Sie Schlüssel **außerhalb** des Codes
# - Nutzen Sie **Umgebungsvariablen** (Environment Variables)
# - In Python mit `os.getenv()` abrufen

# %%
from json import load
import os

# %% [markdown]
#
# ## Umgebungsvariablen mit `os.getenv()`

# %%

# %%

# %% [markdown]
#
# - Wenn die Variable nicht gesetzt ist, gibt `os.getenv()` `None` oder einen
#   Defaultwert zurück
# - Wir müssen die Variable noch setzen!

# %% [markdown]
#
# ## Die `.env` Datei
#
# - Erstellen Sie eine Datei namens `.env` im Projektordner
# - Speichern Sie dort Ihre Schlüssel:
#
# ```bash
# # .env Datei
# OPENROUTER_API_KEY=sk-or-v1-abc123...
# ```
#
# **Wichtig:** `.env` niemals in Git committen!

# %% [markdown]
#
# ## `.gitignore` nicht vergessen!
#
# Fügen Sie `.env` zu Ihrer `.gitignore` Datei hinzu:
#
# ```
# # .gitignore
# .env
# ```
#
# So wird die Datei nie versehentlich hochgeladen.

# %% [markdown]
#
# ## python-dotenv: `.env` laden
#
# Das Package `python-dotenv` lädt `.env` automatisch:

# %%
# !pip install python-dotenv

# %%
from dotenv import load_dotenv

# %% [markdown]
#
# ## `.env` Datei laden

# %%

# %% [markdown]
#
# Um eine andere Datei als `.env` zu verwenden

# %%

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung: API-Schlüssel einrichten
#
# 1. Erstellen Sie eine `.env` Datei:
#    ```
#    OPENROUTER_API_KEY=sk-or-v1-...
#    ```
# 2. Fügen Sie `.env` zu `.gitignore` hinzu
# 3. Laden Sie die Datei in Python:
#    ```python
#    from dotenv import load_dotenv
#    import os
#
#    load_dotenv()
#    api_key = os.getenv("OPENROUTER_API_KEY")
#    ```

# %% [markdown]
#
# ## Erster API-Aufruf: Verfügbare Modelle
#
# - Bevor wir ein LLM nutzen, schauen wir uns die verfügbaren Modelle an
# - OpenRouter bietet einen Endpunkt `/api/v1/models`
# - Dieser zeigt alle verfügbaren Modelle

# %%
import requests
from dotenv import load_dotenv
import os

# %%
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
assert api_key is not None, "API key not set!"

# %% [markdown]
#
# ## Modelle abrufen
#
# Siehe auch: [OpenRouter API
# Dokumentation](https://openrouter.ai/docs/api/api-reference/models/get-models)

# %%
url = "https://openrouter.ai/api/v1/models"

# %%
headers = {"Authorization": f"Bearer {api_key}"}


# %%

# %%

# %% [markdown]
#
# ## Die Antwort untersuchen

# %%

# %%

# %%

# %% [markdown]
#
# ## Ein einzelnes Modell ansehen

# %%

# %% [markdown]
#
# ## Wichtige Felder eines Modells
#
# - `id`: Der Name, den wir in API-Aufrufen verwenden
# - `name`: Menschenlesbarer Name
# - `pricing`: Kosten pro Token
# - `context_length`: Maximale Kontextlänge in Token
# - `description`: Kurze Beschreibung des Modells

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Modelle filtern

# %%
def print_models(models, limit=5):
    for model in models[:limit]:
        print(f"- {model['id']}: {model['name']}")

# %% [markdown]
#
# Einige OpenAI-Modelle:

# %%

# %% [markdown]
#
# Einige Anthropic-Modelle (Claude):

# %%

# %% [markdown]
#
# Kostenlose Modelle:

# %%

# %%

# %%

# %%


# %% [markdown]
#
# ## Was haben wir gelernt?
#
# 1. **API-Schlüssel** sind wie Passwörter für APIs
# 2. **Niemals** Schlüssel im Code speichern
# 3. `.env` Dateien und `python-dotenv` nutzen
# 4. Der erste API-Aufruf zeigt verfügbare Modelle
# 5. OpenRouter bietet Zugang zu vielen verschiedenen LLMs

# %% [markdown]
#
# ## Mini-Workshop: API-Schlüssel einrichten
#
# **Aufgaben:**
#
# 1. Erstellen Sie eine `.env` Datei in Ihrem Projektordner
# 2. Fügen Sie Ihren OpenRouter API-Schlüssel hinzu:
#    ```
#    OPENROUTER_API_KEY=sk-or-v1-...
#    ```
# 3. Laden Sie den Schlüssel mit `python-dotenv`
# 4. Rufen Sie die Modell-Liste von OpenRouter ab
# 5. Finden Sie alle Modelle eines bestimmten Anbieters (z.B. "meta-llama")

# %%
from dotenv import load_dotenv
import requests
import os

# %% [markdown]
#
# ### Lösung

# %%

# %%

# %% [markdown]
#
# Meta Llama Modelle finden:

# %%

# %%

# %%
