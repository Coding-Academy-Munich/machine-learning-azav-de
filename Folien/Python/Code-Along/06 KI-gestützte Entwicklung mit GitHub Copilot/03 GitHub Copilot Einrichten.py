# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>GitHub Copilot Einrichten</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Ziel dieser Lektion
#
# - GitHub Copilot in VS Code installieren
# - Konto verbinden und Subscription wählen
# - Wichtige Einstellungen konfigurieren
# - Installation verifizieren

# %% [markdown]
#
# ## Voraussetzungen
#
# **Benötigte Software:**
# - VS Code installiert (`code.visualstudio.com`)
# - Git installiert (`git-scm.com`)
#
# **Benötigte Konten:**
# - GitHub-Account (`github.com`)

# %% [markdown]
#
# ## Schritt 1: Copilot vorinstalliert (Neuere VS Code Version)
#

# %% [markdown]
#
# ## Schritt 1: Extension installieren (Ältere VS Code Version)
#
# 1. VS Code öffnen
# 2. Extensions-View öffnen (`Ctrl+Shift+X`)
# 3. "GitHub Copilot" suchen
# 4. Auf "Install" klicken
#
# **Hinweis:** Copilot Chat wird automatisch mit installiert

# %% [markdown]
#
# ## Schritt 2: Mit GitHub anmelden
#
# 1. Je nach Version:
#    - Neuere Version: Copilot-Icon in der Statusleiste unten rechts
#    - Ältere Version: Nach Installation erscheint ein Pop-up
# 2. Auf "Use AI Features"/"Sign in to GitHub" klicken
# 3. Browser öffnet sich automatisch
# 4. GitHub-Anmeldedaten eingeben
# 5. Autorisierung bestätigen

# %% [markdown]
#
# ## Schritt 3: Subscription wählen
#
# **Verfügbare Optionen:**
# - **Free Tier**: 2.000 Completions/Monat
#   - Gut zum Ausprobieren
# - **Pro** ($10/Monat): Unbegrenzte Nutzung
#   - Für regelmäßige Nutzung empfohlen
# - **Enterprise**: Für Teams/Unternehmen
#
# **Für diesen Kurs:** Free Tier ist ausreichend

# %% [markdown]
#
# ## Schritt 4: Einstellungen überprüfen
#
# **Zugriff:** Settings → Extensions → GitHub Copilot
#
# **Empfohlene Einstellungen:**
# - `GitHub > Copilot: Enable` → Aktiviert für Python
# - `Editor: Inline Suggest` → Ein
# - `GitHub > Copilot > Editor: Enable Code Actions` → Ein

# %% [markdown]
#
# ## Wichtige Tastenkombinationen
#
# **Merken Sie sich diese:**
# - `Tab`: Vorschlag annehmen
# - `Esc`: Vorschlag ablehnen
# - `Alt+]` / `Alt+[`: Nächster/vorheriger Vorschlag
# - `Ctrl+I`: Inline Chat öffnen
# - `Ctrl+Alt+I`: Copilot Chat Sidebar öffnen

# %% [markdown]
#
# ## Schritt 5: Installation testen
#
# 1. Neue Python-Datei erstellen: `test_copilot.py`
# 2. Folgenden Kommentar eingeben:
# 3. 2-3 Sekunden warten
# 4. Copilot sollte einen Vorschlag anzeigen (grauer Text)
# 5. Mit `Tab` akzeptieren

# %%
# Test comment - type this and wait for Copilot:
# Function to check if a number is even

# %% [markdown]
#
# ## Troubleshooting
#
# **Copilot schlägt nichts vor?**
# - Prüfe: Extension aktiviert? (Status Bar Icon)
# - Prüfe: Richtig angemeldet?
# - Warte 2-3 Sekunden nach Eingabe
# - Neustart von VS Code versuchen
#
# **Status Bar:** Copilot-Icon unten rechts
# - Klick: Aktivieren/Deaktivieren

# %% [markdown]
#
# ## Zusammenfassung
#
# - GitHub Copilot Extension installiert
# - Mit GitHub-Konto verbunden
# - Free Tier oder Pro Subscription gewählt
# - Wichtige Einstellungen konfiguriert
# - Installation getestet
#
# **Nächste Schritte:** Einführung in Copilot
