# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Copilot Agenten (Chat Modi)</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Die vier Agenten/Chat-Modi
#
# **Copilot bietet vier verschiedene Modi:**
#
# | Modus | Beschreibung | Anwendung |
# |-------|--------------|-----------|
# | **Ask** | Fragen stellen, keine Änderungen | Erklärungen, Konzepte |
# | **Edit** | Sie wählen Dateien, Copilot ändert | Gezielte Refactorings |
# | **Agent** | Copilot findet Dateien autonom | Komplexe Features |
# | **Plan** | Erst planen, dann implementieren | Große Änderungen |

# %% [markdown]
#
# ## Modus wechseln
#
# **Im Copilot Chat Panel:**
# - Dropdown-Menü oben im Chat
# - Oder Tastenkombinationen
#
# **Wann welchen Modus?**
# - **Ask**: "Erkläre mir..." / "Was bedeutet..."
# - **Edit**: "Ändere diese 3 Dateien..."
# - **Agent**: "Implementiere Feature X..."
# - **Plan**: "Plane die Implementierung von..."

# %% [markdown]
#
# ## Ask Mode: Fragen und Lernen
#
# **Eigenschaften:**
# - Keine Code-Änderungen
# - Nur Erklärungen und Antworten
# - Gut für Verständnisfragen
#
# **Beispiele:**
# - "Erkläre den Unterschied zwischen async und await"
# - "Was macht diese Regex?"
# - "Wie funktioniert der Cache in diesem Projekt?"

# %% [markdown]
#
# ## Edit Mode: Kontrollierte Änderungen
#
# **Eigenschaften:**
# - Sie wählen explizit die Dateien
# - Präzise Kontrolle
# - Diff-Vorschau vor Anwendung
#
# **Workflow:**
# 1. Dateien zum Kontext hinzufügen
# 2. Änderung beschreiben
# 3. Diffs reviewen
# 4. Akzeptieren oder ablehnen

# %% [markdown]
#
# ## Agent Mode: Autonomes Arbeiten
#
# **Eigenschaften:**
# - Copilot findet relevante Dateien selbst
# - Kann Terminal-Befehle ausführen
# - Selbstkorrektur bei Fehlern
# - Iterativer Prozess
#
# **Gut für:**
# - Neue Features über mehrere Dateien
# - Refactorings mit unklarem Scope
# - Test-Generierung

# %% [markdown]
#
# ## Plan Mode: Erst denken, dann handeln
#
# **Eigenschaften:**
# - Erstellt detaillierten Implementierungsplan
# - Zeigt betroffene Dateien und Änderungen
# - Plan kann überprüft und angepasst werden
# - Erst nach Freigabe: Implementierung
#
# **Ideal für:**
# - Große, komplexe Änderungen
# - Architektur-Entscheidungen
# - Wenn Überblick wichtig ist

# %% [markdown]
#
# ## Vergleich: Wann welchen Modus?
#
# | Situation | Empfohlener Modus |
# |-----------|-------------------|
# | "Erkläre diesen Code" | Ask |
# | "Füge Docstrings zu diesen 2 Dateien hinzu" | Edit |
# | "Implementiere Caching für die API" | Agent |
# | "Plane eine neue Authentifizierung" | Plan |
# | Schnelle Frage zwischendurch | Ask |
# | Unbekannter Scope der Änderung | Agent oder Plan |

# %% [markdown]
#
# ## Praxis: Plan Mode mit `CopilotAdvancedDemo`
#
# **Aufgabe:** Neues Feature planen
#
# 1. Öffnen Sie `CopilotAdvancedDemo`
# 2. Wählen Sie **Plan** Modus
# 3. Prompt:
#    ```
#    Plan: Add a weather alerts feature that notifies
#    users when extreme temperatures are forecasted.
#    Consider data models, API changes, and dashboard
#    integration.
#    ```
# 4. Reviewen Sie den generierten Plan
# 5. Passen Sie ggf. an
# 6. Wechseln Sie in den **Agent** Modus zur Implementierung
# 7. Überwachen und steuern Sie den Prozess
