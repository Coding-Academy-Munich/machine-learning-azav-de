# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Chat-Teilnehmer</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Chat-Teilnehmer: Spezialisierte Helfer
#
# **Was sind Chat-Teilnehmer (Chat Participants)?**
# - Spezialisierte Agenten in Copilot Chat
# - Zugriff mit `@` gefolgt vom Namen
# - Jeder hat eigene Fähigkeiten und Kontext
#
# **Verfügbare Teilnehmer:**
# - `@workspace` - Codebase-Kontext
# - `@githubpr` - GitHub-Integration (benötigt GitHub Pull Requests Extension)
# - `@terminal` - Terminal-Kontext
# - `@vscode` - Editor-Einstellungen

# %% [markdown]
#
# ## @workspace: Codebase-Kontext
#
# **Fähigkeiten:**
# - Versteht gesamte Projektstruktur
# - Sucht in allen Dateien
# - Kennt Dependencies und Imports
#
# **Beispiele:**
# ```
# @workspace Where is the main processing logic located?
# @workspace What files use the WeatherAPI class?
# @workspace How does the caching work in this project?
# ```

# %% [markdown]
#
# ## @workspace: Praktische Anwendung
#
# **Öffnen Sie `CopilotAdvancedDemo` und fragen Sie:**
#
# ```
# @workspace Explain the architecture of this weather
# dashboard application. What are the main components?
# ```
#
# **Copilot wird:**
# - Alle relevanten Dateien analysieren
# - Zusammenhänge erkennen
# - Strukturierte Erklärung liefern

# %% [markdown]
#
# ## Weitere @workspace Beispiele
#
# **Probieren Sie in `CopilotAdvancedDemo`:**
#
# ```
# @workspace What files would I need to modify to add
# a new data source for weather information?
# ```
#
# ```
# @workspace Show me all places where WeatherData is used
# ```
#
# ```
# @workspace What's the relationship between dashboard.py
# and data_processor.py?
# ```

# %% [markdown]
#
# ## @githubpr: GitHub-Integration
#
# **Fähigkeiten:**
# - Zugriff auf Issues und PRs
# - Repository-Informationen
# - Commit-Historie
# - GitHub-spezifische Aufgaben
#
# **Beispiele:**
# ```
# @githubpr What are the open issues in this repository? Don't filter by label.
# @githubpr Summarize the last 5 pull requests
# @githubpr What changed in the last release?
# ```

# %% [markdown]
#
# ## @terminal: Terminal-Kontext
#
# **Fähigkeiten:**
# - Versteht Terminal-Ausgaben
# - Hilft bei Befehlen und Fehlern
# - Kennt Terminal/Shell-Kontext
#
# **Beispiele:**
# ```
# @terminal How do I run the tests in this project?
# @terminal Explain this error message: "ModuleNotFoundError: No module named 'requests'"
# @terminal Look at the error logs currently in my terminal and suggest possible fixes.
# @terminal What's the git command to undo the last commit?
# ```

# %% [markdown]
#
# ## @vscode: Editor-Einstellungen
#
# **Fähigkeiten:**
# - VS Code Konfiguration
# - Extension-Empfehlungen
# - Tastenkombinationen
# - Workspace-Einstellungen
#
# **Beispiele:**
# ```
# @vscode How do I enable format on save?
# @vscode What extensions are recommended for Python?
# @vscode How do I change the theme?
# ```

# %% [markdown]
#
# ## Participants kombinieren
#
# **Komplexe Anfragen:**
# ```
# @workspace @githubpr Create an issue for the missing
# error handling in the weather_api.py file
# ```
#
# **Copilot kann:**
# - Workspace analysieren (Problem finden)
# - GitHub-Issue erstellen
# - Kontext aus beiden Quellen nutzen

# %% [markdown]
#
# ## Best Practices
#
# **Wann welchen Participant nutzen:**
# - `@workspace`: Fragen zur Codebase, Architektur
# - `@githubpr`: Issues, PRs, Repository-Management
# - `@terminal`: Befehle, Fehler, Shell-Hilfe
# - `@vscode`: Editor-Konfiguration, Extensions
#
# **Tipp:** Participant weglassen = Copilot wählt automatisch
