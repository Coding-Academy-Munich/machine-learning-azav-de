# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>MCP Server Integration</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Model Context Protocol (MCP)
#
# **Was ist MCP?**
# - Offener Standard für KI-Tool-Integration
# - Entwickelt von Anthropic (November 2024)
# - Übernommen von OpenAI, Google, Microsoft
#
# **Die Analogie:**
# - "USB-C für KI-Anwendungen"
# - Ein universeller Anschluss für alle Tools

# %% [markdown]
#
# ## Warum brauchen wir MCP?
#
# **LLMs sind leistungsstarke Reasoning-Engines, aber...**
#
# - Sie haben nur Zugriff auf ihre Trainingsdaten
# - Sie können nicht:
#   - Ihre Dateien lesen
#   - Auf Datenbanken zugreifen
#   - APIs aufrufen
#   - Mit externen Diensten interagieren
#
# **Für echte Aufgaben brauchen wir externe Tools!**

# %% [markdown]
#
# ## Das Problem: M×N Integrationen
#
# **Ohne MCP:**
# - M KI-Anwendungen × N Tools = M×N Integrationen
# - Jede App braucht eigene Connector für jedes Tool
# - Doppelter Aufwand, inkonsistente Implementierungen
#
# **Mit MCP:**
# - M Anwendungen + N Server = M+N Implementierungen
# - Jede App implementiert MCP einmal
# - Jedes Tool stellt einen MCP-Server bereit

# %% [markdown]
#
# ## MCP Architektur
#
# <img src="img/mcp-servers.png" style="width:50%;margin:auto" alt="MCP Server Architecture"></img>

# %% [markdown]
#
# ## Was MCP-Server bereitstellen
#
# | Typ | Kontrolle | Beschreibung |
# |-----|-----------|--------------|
# | **Tools** | KI-gesteuert | Aktionen, die das Modell ausführen kann |
# | **Resources** | App-gesteuert | Kontextdaten für das Modell |
# | **Prompts** | User-gesteuert | Vordefinierte Interaktionen |
#
# **Beispiele:**
# - Tool: "Erstelle ein GitHub Issue"
# - Resource: "Inhalt der Datei config.json"
# - Prompt: "Analysiere diesen Code auf Sicherheit"

# %% [markdown]
#
# ## Verfügbare MCP-Server
#
# **Offizielle Server:**
# - **GitHub** - Repository, Issues, PRs, Actions
# - **Filesystem** - Dateizugriff mit Pfadbeschränkungen
# - **Git** - Repository-Operationen
# - **Memory** - Persistenter Wissensgraph
# - **Fetch** - Web-Inhalte abrufen
#
# **Community-Server:**
# - Datenbanken: PostgreSQL, MongoDB, Redis
# - Cloud: AWS, Azure, Google Cloud
# - Tausende Server auf `github.com/punkpeye/awesome-mcp-servers`

# %% [markdown]
#
# ## Der GitHub MCP-Server
#
# **Besonderheiten:**
# - Remote gehostet (kein Docker nötig)
# - OAuth-Authentifizierung
# - Automatische Updates
#
# **Funktionen:**
# - Repository-Suche und Code-Navigation
# - Issues und Pull Requests erstellen/bearbeiten
# - CI/CD-Workflows analysieren
# - Sicherheitswarnungen abrufen

# %% [markdown]
#
# ## MCP in VS Code konfigurieren
#
# **Konfigurationsdatei:** `.vscode/mcp.json`
#
# ```json
# {
#   "servers": {
#     "server-name": {
#       "command": "npx",
#       "args": ["-y", "@package/server"],
#       "env": { "API_KEY": "${input:api-key}" }
#     }
#   },
#   "inputs": [
#     {
#       "type": "promptString",
#       "id": "api-key",
#       "description": "API Key",
#       "password": true
#     }
#   ]
# }
# ```

# %% [markdown]
#
# ## Transport-Typen
#
# **stdio (Standard I/O):**
# - Für lokale Server
# - Kommunikation über stdin/stdout
# - Server läuft als Child-Prozess
#
# **HTTP/SSE:**
# - Für Remote-Server
# - Kommunikation über Netzwerk
# - Server läuft extern (Cloud)
#
# ```json
# {
#   "github-remote": {
#     "type": "http",
#     "url": "https://api.githubcopilot.com/mcp/"
#   }
# }
# ```

# %% [markdown]
#
# ## GitHub MCP-Server installieren
#
# **Einfachste Methode:**
# 1. Command Palette öffnen (`Ctrl+Shift+P`)
# 2. `GitHub MCP: Install Remote Server` ausführen
# 3. OAuth-Authentifizierung durchführen
# 4. VS Code neu starten
#
# **Der Server ist dann automatisch verfügbar!**
#
# **Alternativ:** One-Click Install Button auf GitHub Docs

# %% [markdown]
#
# ## Demo: GitHub MCP-Server verwenden
#
# **Beispiel-Prompts im Copilot Chat:**
#
# ```
# Create an issue about the login bug with high priority
# ```
#
# ```
# Why did the CI workflow fail last night?
# ```
#
# ```
# List all open pull requests in this repository
# ```
#
# ```
# Show me Dependabot alerts for this repo
# ```

# %% [markdown]
#
# ## Praktischer Workflow
#
# **Komplettes Beispiel:**
#
# 1. **Fehler finden:**
#    `"Show me why the tests failed in the last CI run"`
#
# 2. **Problem analysieren:**
#    `"What's causing the timeout in test_api.py?"`
#
# 3. **Issue erstellen:**
#    `"Create an issue for this test failure with the error details"`
#
# 4. **Fix implementieren und PR öffnen:**
#    `"Open a draft PR with my current changes"`

# %% [markdown]
#
# ## Lokalen MCP-Server konfigurieren
#
# **Beispiel: Filesystem-Server**
#
# ```json
# {
#   "servers": {
#     "filesystem": {
#       "command": "npx",
#       "args": [
#         "-y",
#         "@modelcontextprotocol/server-filesystem",
#         "/path/to/allowed/directory"
#       ]
#     }
#   }
# }
# ```
#
# **Sicherheit:** Nur angegebene Pfade sind zugänglich!

# %% [markdown]
#
# ## Demo: Filesystem-Server in Aktion
#
# **Unser Demo-Szenario:**
#
# ```
# mcp_demo/appliances/
# ├── turbochef_3000.md      # Smarter Backofen
# ├── aquapure_pro.md        # Wasserfilter
# └── frostmaster_elite.md   # Kühlschrank
# ```
#
# **Diese Produkte existieren nicht!**
# - Das Modell kann sie nicht aus seinen Trainingsdaten kennen
# - Erst mit dem Filesystem-Server werden sie zugänglich

# %% [markdown]
#
# ## Demo: Konfiguration
#
# **`.vscode/mcp.json`:**
#
# ```json
# {
#   "servers": {
#     "appliances": {
#       "command": "npx",
#       "args": [
#         "-y",
#         "@modelcontextprotocol/server-filesystem",
#         "./mcp_demo/appliances"
#       ]
#     }
#   }
# }
# ```
#
# Der Server hat nur Zugriff auf den `appliances` Ordner!

# %% [markdown]
#
# ## Demo: Beispiel-Prompts
#
# **Mit dem Filesystem-Server können wir jetzt fragen:**
#
# ```
# Welche Energieeffizienzklassen haben die Geräte?
# ```
#
# ```
# Welches Gerät ist am teuersten?
# ```
#
# ```
# Vergleiche den TurboChef 3000 mit dem FrostMaster Elite.
# ```
#
# **Ohne den Server:** Das Modell müsste raten oder ablehnen

# %% [markdown]
#
# ## Sicherheitsaspekte
#
# **MCP-Server können viel Macht haben!**
#
# **Risiken:**
# - Zugriff auf sensible Daten
# - Ausführung von Aktionen (Issues, PRs, Commits)
# - Prompt Injection über Tool-Ergebnisse
#
# **Schutzmaßnahmen:**
# - Nur vertrauenswürdige Server verwenden
# - Read-Only Modus aktivieren wenn möglich
# - Minimale Berechtigungen konfigurieren
# - Server-Logs überprüfen

# %% [markdown]
#
# ## Read-Only und selektive Toolsets
#
# **Read-Only Modus (GitHub MCP):**
# ```json
# {
#   "headers": {
#     "X-MCP-Readonly": "true"
#   }
# }
# ```
#
# **Selektive Toolsets:**
# ```json
# {
#   "headers": {
#     "X-MCP-Toolsets": "context,issues,pull_requests"
#   }
# }
# ```
#
# **Damit:** Nur Lesen, nur bestimmte Aktionen erlaubt

# %% [markdown]
#
# ## Best Practices
#
# **Server-Auswahl:**
# - Offizielle/verifizierte Server bevorzugen
# - Community-Server sorgfältig prüfen
# - Source Code reviewen wenn möglich
#
# **Konfiguration:**
# - Read-Only Modus für Exploration
# - Minimale Berechtigungen
# - Secrets nie in der Config hardcoden
#
# **Betrieb:**
# - Server regelmäßig aktualisieren
# - Logs monitoren
# - Bei Problemen: Server deaktivieren

# %% [markdown]
#
# ## Zusammenfassung
#
# - **MCP:** Universeller Standard für KI-Tool-Integration
# - **Architektur:** Host → Clients → Server (JSON-RPC)
# - **Server bieten:** Tools, Resources, Prompts
# - **GitHub MCP:** Repository-Automatisierung ohne IDE-Wechsel
# - **Konfiguration:** `.vscode/mcp.json`
# - **Sicherheit:** Read-Only, selektive Toolsets, nur vertrauenswürdige Server
