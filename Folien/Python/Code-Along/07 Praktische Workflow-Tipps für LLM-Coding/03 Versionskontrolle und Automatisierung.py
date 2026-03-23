# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Versionskontrolle und Automatisierung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Git als Sicherheitsnetz
#
# - **Häufig committen** während KI-Sessions
# - Aussagekräftige Commit-Messages für intelligentes Revert
# - Niemals KI lange auf uncommitteten Änderungen arbeiten lassen
# - **Regel:** Jeder erfolgreiche Schritt = ein Commit

# %% [markdown]
#
# ## Commit-Strategie während KI-Sessions
#
# ```bash
# # Nach erfolgreichem Plan-Review
# git commit -m "Plan: Add notification system"
#
# # Nach Implementierung von Teil 1
# git commit -m "Add NotificationService"
# ```
#
# ```bash
# # Nach Implementierung von Teil 2
# git commit -m "Add EmailNotifier"
#
# # Wenn KI Fehler macht...
# git checkout -- .  # oder
# git reset --hard HEAD~1
# ```

# %% [markdown]
#
# ## Git Worktrees für parallele Sessions
#
# - Mehrere Claude/Copilot Sessions von verschiedenen Worktrees ausführen
# - **Niemals** Agents gleichzeitig im selben Verzeichnis laufen lassen
# - Vermeidet Konflikte und unvorhersehbares Verhalten

# %% [markdown]
#
# ## Git Worktrees einrichten
#
# ```bash
# # Worktree für Feature A erstellen
# git worktree add -b feature-a ../project-feature-a
#
# # In neuem Terminal: Claude für Feature A
# cd ../project-feature-a
# claude
#
# # Im Original-Verzeichnis: Claude für Feature B
# git checkout -b feature-b
# claude
# ```
#
# Beide Sessions arbeiten unabhängig voneinander!

# %% [markdown]
#
# ## Checkpoints und Rewind (Claude Code)
#
# - Claude Code erstellt automatisch **Checkpoints**
# - `/rewind` um sicher zurückzugehen
# - Alternativen erkunden ohne Fortschritt zu verlieren
# - **Escape-Taste** unterbricht und erhält Kontext

# %% [markdown]
#
# ## Checkpoints bei Copilot
#
# - Copilot hat **keine eingebauten Checkpoints**
# - Git-Commits sind Ihr Sicherheitsnetz
# - **Noch häufiger committen** als bei Claude Code
# - IDE Undo/Redo für kleine Änderungen nutzen
# - Bei größeren Fehlern: `git checkout -- .`

# %% [markdown]
#
# ## Automatisierung zuerst einrichten
#
# **Vor** KI-gestützter Arbeit etablieren:
#
# - **Formatter:** Black, Prettier, etc.
# - **Linter:** Ruff, ESLint, etc.
# - **Type Checker:** mypy, TypeScript
# - **Pre-commit Hooks:** Automatische Prüfungen
# - **CI Pipeline:** Tests bei jedem Push

# %% [markdown]
#
# ## Python-Projekt einrichten (Beispiel)
#
# ```bash
# # Tools installieren
# pip install black ruff mypy pre-commit
# ```
#
# ```bash
# # Pre-commit Konfiguration (.pre-commit-config.yaml)
# repos:
#   - repo: https://github.com/astral-sh/ruff-pre-commit
#     rev: v0.4.10
#     hooks:
#       - id: ruff
#         args: [--fix, --exit-non-zero-on-fix]
#       - id: ruff-format
# ```
#
# ```bash
# # Hooks aktivieren
# pre-commit install
# ```

# %% [markdown]
#
# ## KI zum Ausführen der Tools anleiten
#
# - Explizit sagen: "Führe Formatter, Linter und Tests aus bevor du fertig bist"
# - Befehle in Konfigurationsdatei dokumentieren:
#   - **Claude Code:** `CLAUDE.md`
#   - **Copilot:** `.github/copilot-instructions.md`
# - KI Linting/Type-Fehler selbst beheben lassen

# %% [markdown]
#
# ## Konfigurationsdatei-Beispiele
#
# **CLAUDE.md:** oder **.github/copilot-instructions.md:**
#
# ```markdown
# ### Nach jeder Änderung ausführen:
# - `black .` - Code formatieren
# - `ruff check --fix .` - Linting
# - `mypy src/` - Type Checking
# - `pytest` - Tests
# ```

# %% [markdown]
#
# ## Beispielprojekt: Board Game Night Organizer
#
# Ein vollständig eingerichtetes Projekt das alle Best Practices demonstriert:
#
# - **uv** für Paketmanagement
# - **ruff** für Linting und Formatierung
# - **mypy** (strict mode) für Type Checking
# - **pytest** für Tests
# - **pre-commit** Hooks
# - **CLAUDE.md** mit Tooling-Anweisungen
# - **Claude Code Hooks** für automatische Formatierung

# %% [markdown]
#
# ## Projektstruktur
#
# ```text
# BoardgameNight/
# ├── src/boardgame_night/    # Hauptpaket
# ├── tests/                  # Pytest Tests
# ├── docs/                   # Feature-Implementierungsguide
# ├── .claude/settings.json   # Claude Code Hooks
# ├── .pre-commit-config.yaml # Pre-commit Konfiguration
# ├── CLAUDE.md               # KI-Assistenten-Anweisungen
# └── pyproject.toml          # Projekt-Konfiguration
# ```
#
# Dieses Projekt ist im Kurs-Repository unter `Code/BoardgameNight/` verfügbar.

# %% [markdown]
#
# ## Der professionelle KI-Workflow
#
# 1. **Planung:** Plan Mode, Review vor Implementierung
# 2. **Testing:** TDD, Test-Änderungen überwachen
# 3. **Versionskontrolle:** Häufige Commits, Worktrees
# 4. **Automatisierung:** Formatter, Linter, CI als Guardrails
#
# Diese vier Säulen ermöglichen nachhaltige KI-gestützte Entwicklung!

# %% [markdown]
#
# ## Workshop: Vollständiges Workflow-Setup
#
# **Teil 1: Projekt mit Tooling einrichten**
#
# 1. Neues Python-Projekt erstellen
# 2. Virtuelle Umgebung einrichten
# 3. Installieren: `black`, `ruff`, `mypy`, `pytest`, `pre-commit`
# 4. `.pre-commit-config.yaml` erstellen
# 5. `pre-commit install` ausführen
# 6. Konfigurationsdatei erstellen (`CLAUDE.md` oder `.github/copilot-instructions.md`)

# %% [markdown]
#
# ## Workshop: Vollständiges Workflow-Setup (Fortsetzung)
#
# **Teil 2: Git Worktree erstellen**
#
# ```bash
# # Initialen Commit machen
# git init
# git add .
# git commit -m "Initial project setup"
# ```
#
# ```bash
# # Feature-Branch und Worktree erstellen
# git worktree add ../projekt-feature-a -b feature-a
#
# # Verifizieren
# ls ../projekt-feature-a
# ```

# %% [markdown]
#
# ## Workshop: Vollständiges Workflow-Setup (Fortsetzung)
#
# **Teil 3: KI-Feature mit vollem Workflow**
#
# 1. Claude Code/Copilot im Worktree starten
#    - Claude Code: `claude` im Terminal
#    - Copilot: Ordner in VS Code öffnen
# 2. Feature anfragen (Claude: Plan Mode nutzen)
# 3. Plan reviewen und genehmigen
# 4. Nach jedem Schritt committen
# 5. Sicherstellen dass KI Formatter/Linter ausführt
# 6. Tests nach jeder Änderung laufen lassen

# %% [markdown]
#
# ## Workshop: Vollständiges Workflow-Setup (Fortsetzung)
#
# **Teil 4: Revert üben**
#
# 1. Absichtlich schlechte Änderung von KI anfordern
# 2. Mit `git log --oneline` Commits anzeigen
# 3. Mit `git reset --hard HEAD~1` zurückgehen
# 4. Oder: Spezifische Datei zurücksetzen mit `git checkout HEAD~1 -- datei.py`
# 5. Diskutieren: Wann welche Methode?
