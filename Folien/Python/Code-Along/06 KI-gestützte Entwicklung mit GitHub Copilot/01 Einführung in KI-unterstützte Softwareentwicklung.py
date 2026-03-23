# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in KI-unterstützte Softwareentwicklung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Willkommen zur KI-unterstützten Softwareentwicklung
#
# - Moderne Entwicklung nutzt AI-Tools für höhere Produktivität
# - Tools wie GitHub Copilot, Claude Code und Google Antigravity verändern die Arbeitsweise
# - Wichtig: Realistische Erwartungen und kritisches Denken
# - In diesem Kurs: Praktische Fähigkeiten für den Einsatz von AI im Entwicklungsalltag

# %% [markdown]
#
# ## Die Evolution der Entwickler-Tools
#
# - **1990s**: Einfache Text-Editoren
# - **2000s**: IDEs mit IntelliSense (Autovervollständigung)
# - **2010s**: Intelligente Code-Analyse und Refactoring
# - **2020s**: AI-basierte Code-Generierung
# - **2025**: AI-Agenten und multi-modale Entwicklung

# %% [markdown]
#
# ## Typen von AI-Unterstützung
#
# - **Code-Vervollständigung**: Vorschläge während des Tippens (GitHub Copilot)
# - **Code-Generierung**: Komplette Funktionen/Klassen aus Beschreibung
# - **Chat-Assistenten**: Fragen stellen und Erklärungen erhalten
# - **AI-Agenten**: Autonome mehrstufige Aufgabenerledigung (Claude Code, Antigravity)

# %% [markdown]
#
# ## Überblick der wichtigsten Tools (2025)
#
# - **GitHub Copilot**: Inline-Vorschläge, Chat, mehrere Modelle (GPT-4, Claude, Gemini)
# - **Claude Code**: Terminal-native, 200K Token Context, tiefes Codebase-Verständnis
# - **Google Antigravity**: Neu! Multi-Pane agentic coding mit Gemini 3
# - **Cursor**: Standalone AI-IDE, gebaut auf VS Code
# - **Gemini Code Assist**: Agent-Modus, Next Edit Predictions

# %% [markdown]
#
# ## Was AI-Tools GUT können (2025)
#
# - ✅ Boilerplate-Code und Standard-Patterns
# - ✅ Einfache Algorithmen und Datenmanipulation
# - ✅ Test-Gerüste generieren
# - ✅ Code erklären und dokumentieren
# - ✅ Debuggen und Fehler beheben
# - ✅ Refactoring-Vorschläge
# - ✅ Routine-Aufgaben beschleunigen
# - ✅ Programme für bestimmte Aufgaben schreiben

# %% [markdown]
#
# ### Beispiel
#
# <pre style="font-size: 60%; line-height: 1.2; background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
# I want to create a breakout game in Python.
# Investigate which libraries I can use for this game.
# The game should start with a splash screen before showing the main area.
# The paddle should be controllable via mouse or keyboard. It should be easy
# to add new sprites for bricks.
# When a round ends, the game should show a message to the player indicating
# that they won or lost. It should prompt the player whether they want to quit
# or play again.
# Create a document with requirements, design, and implementation choices.
# </pre>

# %% [markdown]
#
# <pre style="font-size: 60%; line-height: 1.2; background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
# When running the game, I get the following error:
#
# ```
# Breakout> uv run main.py
# Traceback (most recent call last):
#   File "C:\Users\...\Breakout\main.py", line 35, in <module>
#     main()
# ...
#   File "C:\Users\...\Breakout\views\game_view.py", line 71, in on_draw
#     self.paddle.draw()
#     ^^^^^^^^^^^^^^^^
# AttributeError: 'Paddle' object has no attribute 'draw'
# ```
#
# Please investigate the root causes and fix them. Investigate whether similar
# errors are present in other places.
# </pre>

# %%
from IPython.display import Video
Video("img/breakout.mp4")


# %% [markdown]
#
# ## Kritische Einschränkungen (2025)
#
# - ❌ **Sicherheit**: 48% des AI-Codes enthält Schwachstellen
# - ❌ **Halluzinationen**: 20% der Dependencies existieren nicht
# - ❌ **Code-Qualität**: 8x mehr duplizierter Code
# - ❌ **Geschwindigkeit?**: [METR-Studie (Juli 2025)](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/): Erfahrene Entwickler 19% langsamer
# - ❌ **Business-Logik**: Schwaches Verständnis komplexer Domänen
# - ❌ **Vertrauen**: Entwickler vertrauen AI-Code nur "etwas"

# %% [markdown]
#
# ## Die Realität: Produktivität vs. Qualität
#
# - GitHub berichtet: 55% Produktivitätssteigerung
# - ABER: METR-Studie 2025 zeigt komplexeres Bild
# - Entwickler *fühlen* sich 20% schneller
# - Tatsächliche Messungen: 19% langsamer bei erfahrenen Entwicklern
# - **Fazit**: Kritisches Review ist unerlässlich

# %% [markdown]
#
# ## Wann AI NICHT verwenden
#
# - Komplexe Business-Logik mit spezifischem Domänenwissen
# - Sicherheitskritischer Code (Authentication, Authorization)
# - Proprietäre/sensible Projekte (Datenschutz!)
# - Neuartige Algorithmen
# - Beim Lernen neuer Konzepte (Verstehen > Geschwindigkeit)

# %% [markdown]
#
# ## Kursüberblick
#
# **Grundlagen (Dieser Abschnitt):**
# - AI-Tools und LLM-Grundlagen
# - Prompt Engineering
# - Tool-Setup und Best Practices
#
# **Weitere Abschnitte:**
# - AI-unterstützter Entwicklungs-Workflow
# - Code-Qualität mit Clean Code Prinzipien
# - Testing und Sicherheit
# - Praxis-Projekte

# %% [markdown]
#
# ## Kurs-Philosophie
#
# - **Ehrliche Einschätzung**: Fähigkeiten UND Grenzen
# - **Qualität zuerst**: AI beschleunigt, aber Qualität nicht opfern
# - **Human-in-the-Loop**: AI unterstützt, Menschen entscheiden
# - **Praktischer Fokus**: Echte Projekte statt Spielzeug-Beispiele

# %% [markdown]
#
# ## Beispiel: Code-Vervollständigung
#
# Sehen wir uns an, wie AI-Assistenten im Alltag helfen können:

# %%
# Beispiel: Funktion zum Berechnen der Fakultät
# Example: Function to calculate factorial


# %%

# %%

# %%

# %% [markdown]
#
# **Was AI hier tun könnte:**
# - Funktion automatisch vervollständigen nach dem Docstring
# - Tests generieren
# - Iterative Version vorschlagen
# - Edge Cases identifizieren

# %% [markdown]
#
# ## Beispiel: Code-Generierung aus Beschreibung
#
# **Prompt**: "Schreibe eine Funktion, die prüft, ob eine Zahl eine Primzahl ist"

# %%

# %%

# %%

# %%

# %% [markdown]
#
# **Wichtig**: Immer reviewen!
# - Ist die Logik korrekt?
# - Edge Cases behandelt?
# - Performanz akzeptabel?

# %% [markdown]
#
# ## Wichtigste Erkenntnisse
#
# - AI-Tools sind mächtige Helfer, aber kein Ersatz für Entwickler-Wissen
# - Realistische Erwartungen: 48% Sicherheitsprobleme, 19,7% Halluzinationen
# - Kritisches Review ist unerlässlich
# - Produktivität vs. Qualität: Balance finden
# - Nächste Schritte: LLM-Grundlagen verstehen

# %% [markdown]
#
# ## Workshop: Erste Schritte mit AI-Code
#
# Experimentieren Sie mit dem folgenden Szenario:
#
# 1. Überlegen Sie sich eine einfache Funktionalität, die Sie oft schreiben
# 2. Notieren Sie, wie Sie diese normalerweise implementieren würden
# 3. Überlegen Sie, welche Teile AI gut generieren könnte
# 4. Welche Aspekte würden Sie manuell überprüfen?

# %% [markdown]
#
# ## Mini-Workshop: Code Review Übung
#
# Schauen Sie sich den folgenden AI-generierten Code an und identifizieren Sie
# potenzielle Probleme:

# %%
def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = sum(numbers)
    return total / len(numbers)


# %% [markdown]
#
# **Fragen zum Nachdenken:**
# - Was passiert mit einer leeren Liste?
# - Wie werden None-Werte behandelt?
# - Gibt es bessere Implementierungen?

# %% [markdown]
#
# ## Mögliche Verbesserungen

# %%
def calculate_average_safe(numbers):
    """Calculate the average, handling edge cases."""
    # TODO: Add error handling
    pass

# %%
# Tests für die verbesserte Version
# Tests for the improved version

# %%

# %%

# %%
