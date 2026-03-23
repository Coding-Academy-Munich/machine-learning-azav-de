# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Copilot Anpassen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Copilot personalisieren
#
# **Anpassungsmöglichkeiten:**
# - Custom Instructions (Anweisungen)
# - Prompt Files (Wiederverwendbare Vorlagen)
# - Copilot Spaces (Kontextbereiche)
# - MCP Server (Externe Tools)
#
# **Ziel:** Bessere, projektspezifische Vorschläge

# %% [markdown]
#
# ## Custom Instructions
#
# **Was sind Custom Instructions?**
# - Anweisungen, die Copilot bei jeder Anfrage beachtet
# - Definieren Coding-Style, Präferenzen und Projektkontext
# - Werden automatisch zum Prompt hinzugefügt
#
# **Dateitypen:**
# - `.github/copilot-instructions.md` - Hauptdatei für das Repository
# - `.github/instructions/*.instructions.md` - Pfadspezifische Anweisungen
# - `AGENTS.md` / `CLAUDE.md` / `GEMINI.md` - Für Coding Agents

# %% [markdown]
#
# ## Ebenen von Instructions
#
# **Drei Ebenen (nach Priorität):**
#
# 1. **Persönliche Instructions** (höchste Priorität)
#    - Auf GitHub.com konfigurierbar
#    - Gelten für alle Repositories
#
# 2. **Repository Instructions**
#    - Pfadspezifisch → Repository-weit → Agent-Dateien
#    - In diesem Abschnitt behandelt
#
# 3. **Organisations-Instructions** (niedrigste Priorität)
#    - Nur für Copilot Enterprise
#    - Gelten für alle Mitglieder
#
# **Wichtig:** Alle relevanten Instructions werden kombiniert!

# %% [markdown]
#
# ## Beispiel: Custom Instructions
#
# **Datei: `.github/copilot-instructions.md`**
# ```
# # Copilot Instructions
#
# ## Code Style
# - Use type hints for all functions
# - Follow PEP 8 conventions
# - Write docstrings for public APIs
# ```
#
# ```
# ## Preferences
# - Prefer dataclasses over dicts
# - Use pathlib instead of os.path
# - Always include error handling
# ```

# %% [markdown]
#
# ## Pfadspezifische Instructions
#
# - Speicherort: `.github/instructions/`
# - Dateiendung: `*.instructions.md`
# - YAML Frontmatter mit `applyTo` Glob-Pattern
# - Optional: `excludeAgent` zum Ausschließen bestimmter Agents
#
# **Beispiel: `.github/instructions/tests.instructions.md`**
# ```yaml
# ---
# applyTo: "tests/**/*.py"
# excludeAgent: "code-review"
# ---
# - Always use pytest as testing framework
# - Include parametrized tests for edge cases
# - Mock external API calls with pytest-mock
# ```

# %% [markdown]
#
# ## Weitere Beispiele für applyTo
#
# **Frontend-Code:**
# ```yaml
# ---
# applyTo: "src/components/**/*.tsx"
# ---
# - Use React functional components with hooks
# - Include TypeScript interfaces for props
# ```
#
# **API-Endpunkte:**
# ```yaml
# ---
# applyTo: "src/api/**/*.py"
# ---
# - Always validate input parameters
# - Return consistent error responses
# ```

# %% [markdown]
#
# ## Praxis: Instructions testen
#
# **Im `CopilotAdvancedDemo`-Projekt:**
#
# 1. Öffnen Sie `.github/copilot-instructions.md`
# 2. Beachten Sie die definierten Regeln
# 3. Bitten Sie Copilot um eine neue Funktion:
#    ```
#    Add a function to validate city names
#    ```
# 4. Prüfen Sie: Folgt der Code den Instructions?

# %% [markdown]
#
# ## Prompt Files
#
# **Was sind Prompt Files?**
# - Wiederverwendbare Prompt-Vorlagen
# - Für wiederkehrende Aufgaben
# - Per Slash-Command aufrufbar
#
# **Speicherort und Format:**
# - Verzeichnis: `.github/prompts/`
# - Dateiendung: `.prompt.md`
# - Beispiel: `.github/prompts/add-tests.prompt.md`

# %% [markdown]
#
# ## Beispiel: Prompt File
#
# **Datei: `.github/prompts/add-tests.prompt.md`**
# ```yaml
# ---
# description: Add unit tests for the selected code
# agent: agent
# model: Claude Sonnet 4.5 (copilot)
# tools: ['read', 'edit', 'execute', 'search', 'search/codebase', 'web/githubRepo']
# ---
# ```
#
# ```yaml
# Add unit tests for the selected code.
#
# - Use pytest as testing framework.
# - Ensure that the tests cover various edge cases and validate the expected behavior of the code.
# - Apply best practices for writing unit tests in Python.
#
# Reference: #file:tests/conftest.py
# ```

# %% [markdown]
#
# ## Verwenden von Prompt-Dateien
#
# **Aufruf im Chat:**
# - `/` gefolgt vom Dateinamen (ohne `.prompt.md`)
# - Beispiel: `/add-tests` für `add-tests.prompt.md`
#
# **Weitere Optionen:**
# - Command Palette: `Chat: Run Prompt`
# - Play-Button in der geöffneten Prompt-Datei

# %% [markdown]
#
# ## Copilot Spaces
#
# **Was sind Spaces?**
# - Kuratierte Kontextbereiche für Projekte/Aufgaben
# - Bündeln relevante Dateien, Issues, PRs, Docs
# - Bleiben automatisch synchron mit dem Repository
#
# **Mögliche Inhalte:**
# - Repositories, Dateien, Ordner
# - Pull Requests und Issues
# - Freitext-Notizen, Bilder, Uploads

# %% [markdown]
#
# ## MCP Server Integration
#
# **Model Context Protocol (MCP):**
# - Offener Standard für Tool-Integration
# - Erweitert Copilot mit externen Fähigkeiten
# - Beispiele: GitHub, Datenbanken, Dateisystem
#
# **Details:** Eigene Präsentation im nächsten Abschnitt!

# %% [markdown]
#
# ## Settings und Konfiguration
#
# **Erforderliche Einstellungen (müssen aktiviert sein):**
# - `github.copilot.chat.codeGeneration.useInstructionFiles`: true
# - `chat.useAgentsMdFile`: true (für AGENTS.md)
#
# **Weitere Einstellungen:**
# - `github.copilot.enable`: An/Aus pro Sprache
# - Modell-Auswahl: GPT-5.2, Claude Opus 4.5, Gemini 3, ...
# - `chat.promptFilesLocations`: Zusätzliche Prompt-Ordner

# %% [markdown]
#
# ## Dateien erstellen und prüfen
#
# **Dateien per UI erstellen:**
# - Command Palette:
#   - `Chat: Generate Workspace Instructions File`
#   - `Chat: New Instructions File`
# - Chat-Ansicht → Zahnrad → Chat Instructions
#
# **Prüfen ob Instructions verwendet werden:**
# - Im Chat unter "References" nachschauen
# - Dort werden verwendete Instruction-Dateien angezeigt

# %% [markdown]
#
# ## Best Practices
#
# **Für Custom Instructions:**
# - Kurz und präzise halten (keine Romane!)
# - Auf Englisch schreiben
# - Mit Team abstimmen und versionieren
# - Pfadspezifische Instructions für verschiedene Bereiche
#
# **Für Prompt Files:**
# - Für wiederkehrende Aufgaben erstellen
# - Klare Beschreibung im Frontmatter
# - Dateireferenzen für Kontext nutzen
#
# **Wichtig:** Instructions gelten nur für Chat, nicht für Inline-Vorschläge!

# %% [markdown]
#
# ## Übersicht: Datei-Speicherorte
#
# | Dateityp | Speicherort | Dateiendung |
# |----------|-------------|-------------|
# | Custom Instructions | `.github/` | `copilot-instructions.md` |
# | Path-specific | `.github/instructions/` | `*.instructions.md` |
# | Prompt Files | `.github/prompts/` | `*.prompt.md` |
# | Agent Instructions | Beliebig / Wurzel | `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` |

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Custom Instructions:** Automatische Anweisungen für jede Anfrage
# - **Prompt Files:** Wiederverwendbare Vorlagen per Slash-Command
# - **Copilot Spaces:** Kuratierter Projektkontext
# - **MCP Server:** Externe Tools (nächster Abschnitt)
# - Alle Konfigurationsdateien im Repository versionierbar
