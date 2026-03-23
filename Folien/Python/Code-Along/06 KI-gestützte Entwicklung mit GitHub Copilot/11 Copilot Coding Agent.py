# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Copilot Coding Agent</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Copilot Coding Agent
#
# **Was ist der Coding Agent?**
# - Autonomer KI-Entwickler
# - Arbeitet im Hintergrund auf GitHub
# - Erstellt Pull Requests
# - Powered by GitHub Actions
#
# **Unterschied zu Agent Mode:**
# - Agent Mode: Lokal in Ihrer IDE
# - Coding Agent: Remote auf GitHub

# %% [markdown]
#
# ## Wie funktioniert es?
#
# **Workflow:**
# 1. Sie erstellen ein GitHub Issue
# 2. Weisen Sie es `@copilot` zu
# 3. Copilot analysiert das Repository
# 4. Copilot implementiert die Änderungen
# 5. Copilot erstellt einen Pull Request
# 6. Sie reviewen und mergen
#
# **Alles automatisch!**

# %% [markdown]
#
# ## Aufgaben zuweisen
#
# **Option 1: Issue erstellen und zuweisen**
# ```
# Title: Add input validation to user registration
#
# Description:
# - Validate email format
# - Check password strength
# - Return meaningful error messages
#
# Assignee: @copilot
# ```

# %% [markdown]
#
# ## @copilot in Pull Requests
#
# **Option 2: In PR-Kommentaren**
# ```
# @copilot Please add unit tests for the new validation
# functions and fix the type hints.
# ```
#
# **Copilot wird:**
# - Den PR-Kontext verstehen
# - Änderungen direkt zum PR hinzufügen
# - Commit-Messages erstellen

# %% [markdown]
#
# ## Gute Aufgaben für Coding Agent
#
# **Ideal für:**
# - Bug Fixes mit klarer Beschreibung
# - Inkrementelle Features
# - Test-Coverage erhöhen
# - Dokumentation aktualisieren
# - Refactoring nach festem Pattern
#
# **Weniger geeignet:**
# - Architektur-Entscheidungen
# - Komplexe Business-Logik
# - Sicherheitskritische Änderungen

# %% [markdown]
#
# ## Sicherheitsfeatures
#
# **Eingebaute Sicherheit:**
# - Sandbox-Umgebung (GitHub Actions)
# - Kann nur zu `copilot/`-Branches pushen
# - Code wird mit CodeQL geprüft
# - Secret-Scanning aktiv
# - Review durch Menschen erforderlich
#
# **Keine automatischen Merges!**

# %% [markdown]
#
# ## Limitationen
#
# **Aktuelle Einschränkungen:**
# - Nur ein Repository pro Aufgabe
# - Ein PR pro Aufgabe
# - Modell: Claude Sonnet 4.5 (nicht wählbar)
# - Nicht kompatibel mit allen Branch-Regeln
# - Nur GitHub-gehostete Repositories

# %% [markdown]
#
# ## Verfügbarkeit
#
# **Enthalten in:**
# - Copilot Pro
# - Copilot Pro+
# - Copilot Business
# - Copilot Enterprise
#
# **Voraussetzung:**
# - Administrator muss Feature aktivieren
# - Repository-Owner kann opt-out

# %% [markdown]
#
# ## Workflow mit Coding Agent
#
# **Empfohlener Prozess:**
# 1. Klares Issue mit Akzeptanzkriterien erstellen
# 2. `@copilot` zuweisen
# 3. Warten auf PR (Minuten bis Stunden)
# 4. PR gründlich reviewen
# 5. Feedback in Kommentaren geben
# 6. Copilot iteriert
# 7. Mergen wenn zufrieden

# %% [markdown]
#
# ## Integrationen
#
# **Coding Agent außerhalb von GitHub:**
# - **Slack**: Issues per Slack-Nachricht erstellen und zuweisen
# - **Microsoft Teams**: Integration in Team-Workflows
# - **Linear**: Direkte Verbindung zu Linear-Issues
#
# **Vorteil:** Aufgaben aus dem gewohnten Workflow heraus delegieren

# %% [markdown]
#
# ## Zusammenfassung
#
# - Copilot Coding Agent: Autonomer KI-Entwickler
# - Arbeitet auf GitHub im Hintergrund
# - Erstellt PRs aus Issues
# - Integrationen: Slack, Teams, Linear
# - Eingebaute Sicherheitsmaßnahmen
# - Ideal für klar definierte Aufgaben
# - Immer menschliches Review erforderlich
