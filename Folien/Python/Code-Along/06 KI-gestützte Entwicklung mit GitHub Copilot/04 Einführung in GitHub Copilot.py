# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in GitHub Copilot</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Was ist GitHub Copilot?
#
# **AI-Pair-Programmer von GitHub/Microsoft:**
# - Inline Code-Vervollständigung
# - Chat-basierte Unterstützung
# - Multi-Modell-Support (GPT, Claude, Gemini)
# - In VS Code integriert

# %% [markdown]
#
# ## Zwei Hauptfunktionen
#
# **1. Inline-Vorschläge:**
# - Erscheinen automatisch beim Tippen
# - Grauer Text zeigt Vorschläge
# - Tab zum Akzeptieren
#
# **2. Copilot Chat:**
# - Interaktive Konversationen
# - Fragen zu Code stellen
# - Funktionen generieren lassen

# %% [markdown]
#
# ## Wie Inline-Vorschläge funktionieren
#
# **Copilot analysiert:**
# - Aktuellen Datei-Inhalt
# - Kommentare über dem Cursor
# - Funktions- und Variablennamen
# - Andere geöffnete Dateien
# - Imports und Dependencies
#
# **Dann:** Schlägt passenden Code vor

# %% [markdown]
#
# ## Tastenkombinationen im Überblick
#
# | Aktion | Tastenkombination |
# |--------|-------------------|
# | Akzeptieren | `Tab` |
# | Ablehnen | `Esc` |
# | Nächster Vorschlag | `Alt+]` |
# | Vorheriger Vorschlag | `Alt+[` |
# | Nur nächstes Wort | `Ctrl+→` |
# | Inline Chat | `Ctrl+I` |
# | Chat Sidebar | `Ctrl+Alt+I` |

# %% [markdown]
#
# ## Praxis: CopilotStarterKit
#
# **Wir verwenden das CopilotStarterKit:**
# - Ordner: `examples/CopilotStarterKit`
# - Enthält Python-Dateien zum Üben
# - Code zum Erklären lassen
# - Kommentare zum Code generieren
#
# **Öffnen Sie jetzt den Ordner in VS Code!**

# %% [markdown]
#
# ## Übung 1: Code erklären lassen
#
# 1. Öffnen Sie `data_processor.py` aus dem StarterKit
# 2. Markieren Sie eine Funktion
# 3. Öffnen Sie Copilot Chat (`Ctrl+Alt+I`)
# 4. Fragen Sie: "Explain this code"
# 5. Oder nutzen Sie: `/explain`

# %% [markdown]
#
# ## Übung 2: Inline-Vervollständigung
#
# 1. Öffnen Sie `exercises.py` aus dem StarterKit
# 2. Finden Sie die TODO-Kommentare
# 3. Positionieren Sie den Cursor nach dem Kommentar
# 4. Warten Sie auf Copilot-Vorschlag
# 5. Mit `Tab` akzeptieren oder `Alt+]` für Alternativen

# %% [markdown]
#
# ## Übung 3: Copilot Chat nutzen
#
# **Nützliche Chat-Commands:**
# - `/explain`: Code erklären
# - `/fix`: Bugs finden und fixen
# - `/tests`: Tests generieren
# - `/doc`: Dokumentation hinzufügen
#
# **Probieren Sie alle auf Code im StarterKit!**

# %% [markdown]
#
# ## Wann Copilot am besten funktioniert
#
# **Gut für:**
# - Einzelne Funktionen schreiben
# - Boilerplate-Code
# - Standard-Patterns (Loops, Conditionals)
# - Schnelle Vervollständigung
#
# **Weniger gut für:**
# - Komplexe Multi-File-Änderungen
# - Business-Logik-Entscheidungen
# - Sicherheitskritischer Code

# %% [markdown]
#
# ## Wichtig: Nicht blind vertrauen!
#
# - Nicht alle Vorschläge sind korrekt
# - Manche enthalten Sicherheitslücken
# - Andere sind ineffizient oder im Stil unpassend
#
# **Regel:** Immer Code reviewen und testen!

# %% [markdown]
#
# ## Zusammenfassung
#
# - Copilot bietet Inline-Vorschläge und Chat
# - Tab akzeptiert, Alt+] zeigt Alternativen
# - Slash-Commands im Chat für häufige Aufgaben
# - CopilotStarterKit zum Üben nutzen
# - Vorschläge immer kritisch prüfen!
#
# **Nächste Schritte:** Einführung in Claude Code

# %% [markdown]
#
# ## Workshop: Copilot Exploration
#
# **Aufgabe**
# 1. Öffnen Sie CopilotStarterKit in VS Code
# 2. Lassen Sie sich `data_processor.py` erklären
# 3. Vervollständigen Sie TODOs in `exercises.py`
# 4. Generieren Sie Tests für eine Funktion
#
# **Dokumentieren Sie:**
# - Was hat gut funktioniert?
# - Was war überraschend?
# - Hat Copilot Fehler gemacht? Welche?
