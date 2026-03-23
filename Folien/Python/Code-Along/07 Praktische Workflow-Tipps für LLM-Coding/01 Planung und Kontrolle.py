# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Planung und Kontrolle</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Erfolgreich mit KI-Coding-Assistenten arbeiten
#
# - Erfolg erfordert mehr als nur gute Prompts
# - **Vier Schlüsselbereiche:**
#   1. Planung und Kontrolle
#   2. Testing-Strategien
#   3. Versionskontrolle
#   4. Automatisierung

# %% [markdown]
#
# ## Erst planen, dann implementieren
#
# - **Plan Mode verwenden:**
#   - Claude Code: `claude --plan`
#   - Copilot: Explizit nach Plan fragen ("Erstelle erst einen Plan...")
# - Plan reviewen und genehmigen vor der Implementierung
# - **Vorteile:**
#   - Bessere Kontrolle über das Ergebnis
#   - Weniger Überraschungen
#   - Einfachere Kurskorrektur

# %% [markdown]
#
# ## Beispiel: Plan Mode in Aktion
#
# ```
# $ claude --plan
#
# User: Füge ein Benachrichtigungssystem hinzu.
#       Benutzer sollen benachrichtigt werden wenn:
#       - Ein reserviertes Buch verfügbar wird
#       - Ein ausgeliehenes Buch bald fällig ist
# ```
#
# ```
# Claude: Ich erstelle einen Plan für das Benachrichtigungssystem...
#
#         ## Vorgeschlagener Plan:
#         1. NotificationService-Klasse erstellen
#         2. E-Mail-Templates definieren
#         3. Scheduler für Fälligkeitsprüfung
#         4. Integration in ReservationService
# ```
#
# ```
#         Soll ich mit diesem Plan fortfahren?
# ```

# %% [markdown]
#
# ## In kleinen Schritten iterieren
#
# - Komplexe Aufgaben in kleinere Teile aufbrechen
# - Jeden Schritt reviewen bevor es weitergeht
# - **Kontext zurücksetzen:**
#   - Claude Code: `/clear`, Doppel-Escape
#   - Copilot: Neuen Chat starten (`Ctrl+N` / `Cmd+N`)

# %% [markdown]
#
# ## Kontext gezielt verwalten
#
# - Relevante Dateien öffnen, irrelevante schließen
# - **Projekt-Kontext dokumentieren:**
#   - `CLAUDE.md` für Claude Code
#   - `.github/copilot-instructions.md` für Copilot
# - Nicht hilfreiche Chat-Verläufe löschen oder neue Threads starten

# %% [markdown]
#
# ## "Think"-Prompts für komplexe Aufgaben
#
# - Explizites "Denken" auslösen für bessere Ergebnisse
# - **Claude Code Eskalationsstufen:**
#   - `think` - Standard-Nachdenken
#   - `think hard` - Mehr Überlegung
#   - `think harder` - Maximale Tiefe
# - **Copilot:** "Denke Schritt für Schritt" oder ähnliche Prompts
# - Besonders nützlich bei architektonischen Entscheidungen

# %% [markdown]
#
# ## Zusammenfassung: Planung und Kontrolle
#
# | Tipp | Warum |
# |------|-------|
# | Plan Mode | Validiere Ansatz vor Implementierung |
# | Kleine Schritte | Bessere Kontrolle und Review |
# | Kontext verwalten | Fokussierte, relevante Ergebnisse |
# | Think-Prompts | Tiefere Analyse bei komplexen Aufgaben |

# %% [markdown]
#
# ## Workshop: Plan-First Entwicklung
#
# **Wählen Sie ein Projekt:**
# - Idealerweise ein eigenes Projekt, an dem Sie arbeiten möchten
# - Oder eines dieser Beispielprojekte:
#   - Rezept-Manager mit Zutaten-Ersetzungsvorschlägen
#   - Podcast-Zusammenfasser mit Themenextraktion
#   - Pflanzen-Bewässerungsplaner mit Wetter-API-Integration
#
# **Ihre Aufgabe:**
# 1. Plan erstellen lassen (Claude: `--plan`, Copilot: explizit fragen)
# 2. Plan kritisch reviewen
# 3. Änderungen anfordern
# 4. Erst nach Zufriedenheit: Implementierung starten

# %% [markdown]
#
# ## Workshop: Vergleich mit/ohne Plan
#
# **Experiment (optional):**
#
# 1. **Ohne Plan:** Bitten Sie die KI direkt um Implementierung des Features
# 2. **Mit Plan:** Verwenden Sie den Plan-First Ansatz
# 3. **Vergleichen Sie:**
#    - Qualität des Codes
#    - Vollständigkeit der Lösung
#    - Zeit bis zum gewünschten Ergebnis
#    - Anzahl der Nachbesserungen
