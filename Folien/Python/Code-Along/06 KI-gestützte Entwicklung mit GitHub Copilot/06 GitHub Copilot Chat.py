# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>GitHub Copilot Chat</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Copilot Chat
#
# **Zugriff:**
# - Sidebar: Chat-Icon klicken
# - Tastenkombination: `Ctrl+Shift+I`
# - Inline: `Ctrl+I` im Editor
#
# **Verwendung:**
# - Code-Erklärungen
# - Funktions-Generierung
# - Bug-Fixes
# - Refactoring-Vorschläge

# %% [markdown]
#
# ## Interaktion
#
# - Schreiben Sie Prompts in natürlicher Sprache
# - Zeitersparnis durch
#   - Slash Commands (Prompt-Abkürzungen)
#   - Participants (Spezialisten im Chat: wer?)
#   - Agenten (Arbeitsweise: wie?)

# %% [markdown]
#
# ## Slash Commands
#
# **Häufig genutzte Commands:**
# - `/explain`: Code erklären
# - `/fix`: Bugs finden und fixen
# - `/tests`: Tests generieren
# - `/help`: Liste aller Commands anzeigen

# %% [markdown]
#
# ## Beispiel: /explain verwenden
#
# Komplexen Code markieren und `/explain` eingeben.

# %%
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


# %% [markdown]
#
# **Copilot Chat** mit `/explain` erklärt:
# - Divide-and-Conquer Algorithmus
# - Pivot-Element Auswahl
# - Rekursive Partitionierung
# - Zeitkomplexität O(n log n)

# %% [markdown]
#
# ## Inline Chat: Ctrl+I
#
# **Vorteile:**
# - Schneller Zugriff im Editor
# - Kontext der aktuellen Datei
# - Direkte Änderungen am Code
#
# **Workflow:**
# 1. Code markieren (optional)
# 2. `Ctrl+I` drücken
# 3. Anweisung eingeben
# 4. Änderungen reviewen

# %% [markdown]
#
# ### Beispiel
#
# Erzeugen eines Docstrings für die folgende Funktion mit Inline Chat.

# %%
def generate_random_string(
    length: int = 10,
    charset: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
) -> str:
    import random

    return ''.join(random.choice(charset) for _ in range(length))


# %% [markdown]
#
# ## Multi-Modell-Support (2025)
#
# **Verfügbare Modelle (Auswahl):**
# - **Claude Opus 4.5 (Preview)**: Derzeit bestes Modell für Coding Tasks
# - **Claude Sonnet 4.5**: Oft guter Kompromiss zwischen Leistung und Kosten
# - **Claude Haiku 4.5**: Schnell, kostengünstig, gut
# - **GPT-5.1 (Preview)**: Sehr gutes Modell von OpenAI
# - **Gemini 3 Pro (Preview)**: Sehr gutes Modell von Google
#
# **Auswahl:** Chat → Model Selector (oben rechts)

# %% [markdown]
#
# ## Bilder im Chat
#
# **Copilot kann Bilder analysieren:**
# - Screenshots von Fehlern oder UI
# - Diagramme und Mockups
# - Handschriftliche Skizzen
#
# **Verwendung:**
# - Bild per Drag & Drop in Chat
# - Oder: Einfügen aus Zwischenablage
# - Prompt: "Was zeigt dieses Diagramm?" oder "Implementiere dieses UI"

# %% [markdown]
#
# ## Beispiel
#
# Implementiere diese TODO-Listen App in JavaScript mit React. Die Anwendung
# soll ohne Server funktionieren.

# %% [markdown]
#
# <img src="img/ui-sketch.png" alt="UI Sketch" style="width:50%;"/>

# %% [markdown]
#
# ### Ergebnis
#
# - [todo-app/copilot-chat.html](todo-app/copilot-chat.html)
# - [todo-app/index.html](todo-app/index.html)

# %% [markdown]
#
# ## Effektive Chat-Nutzung
#
# **Tipps für bessere Ergebnisse:**
# - Kontext mitgeben (was haben wir schon?)
# - Spezifische Anforderungen nennen (was ist das Ziel?)
# - Bei Fehlern korrigieren/nachfragen
# - Auf vorherige Antworten aufbauen

# %% [markdown]
#
# ## Praxis: Chat mit `CopilotAdvancedDemo`
#
# **Probieren Sie diese Prompts:**
#
# 1. Code verstehen:
#    ```
#    /explain the caching mechanism in weather_api.py
#    ```
#
# 2. Bug fixen:
#    ```
#    /fix potential issues with date handling
#    ```
#
# 3. Tests generieren:
#    ```
#    /tests for the WeatherAPIClient.get_current_weather method
#    ```

# %% [markdown]
#
# ## Zusammenfassung
#
# - Copilot Chat: Interaktive Konversationen
# - Slash-Commands für häufige Aufgaben
# - Inline Chat mit Ctrl+I
# - Multi-Modell-Support
# - Kontext verbessert Ergebnisse
