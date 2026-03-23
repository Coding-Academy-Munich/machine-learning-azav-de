# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Copilot Edits</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Copilot Edits: Multi-File Editing
#
# **Was ist Copilot Edits?**
# - Bearbeitung mehrerer Dateien gleichzeitig
# - Zwei Modi: Edit Mode und Agent Mode
# - Änderungen mit Diff-Vorschau
# - Schrittweise Genehmigung
#
# **Zugriff:** `Chat` → `Edit`

# %% [markdown]
#
# ## Edit Mode vs. Agent Mode
#
# **Edit Mode:**
# - Sie wählen die zu bearbeitenden Dateien
# - Präzise Kontrolle über Änderungen
# - Gut für gezielte Refactorings
#
# **Agent Mode:**
# - Copilot wählt relevante Dateien selbst
# - Autonome Ausführung
# - Gut für komplexe Features

# %% [markdown]
#
# ## Edit Mode: Workflow
#
# **Schritt 1:** Dateien hinzufügen
# - Dateien per Drag & Drop
# - Oder: "Add Files" Button
# - Oder: `#file` im Prompt
#
# **Schritt 2:** Änderung beschreiben
# ```
# Add error handling to all API calls in these files
# ```
#
# **Schritt 3:** Änderungen reviewen und akzeptieren

# %% [markdown]
#
# ## Praxis: Edit Mode
#
# **Öffnen Sie `CopilotAdvancedDemo`:**
#
# 1. Wählen Sie `Copilot` → `Edit`
# 2. Fügen Sie `weather_api.py` und `dashboard.py` hinzu
# 3. Prompt:
#    ```
#    Add proper error handling with try/except blocks
#    to all methods that could raise exceptions
#    ```
# 4. Reviewen Sie die Diff-Vorschau

# %% [markdown]
#
# ## Agent Mode: Autonomes Arbeiten
#
# **Aktivierung:**
# - Edits Panel → Toggle "Agent Mode"
# - Oder: Prompt mit komplexer Aufgabe
#
# **Agent Mode kann:**
# - Relevante Dateien selbst finden
# - Terminal-Befehle ausführen
# - Externe Tools aufrufen
# - Bei Fehlern selbst korrigieren

# %% [markdown]
#
# ## Praxis: Agent Mode
#
# **Aufgabe für Agent Mode:**
#
# ```
# Add a new feature: temperature unit conversion.
# - Add a convert_temperature function to data_processor.py
# - Update the dashboard to support both Celsius and Fahrenheit
# - Add tests for the new functionality
# ```
#
# **Beobachten Sie:** Copilot findet selbst die relevanten Dateien

# %% [markdown]
#
# ## Änderungen reviewen
#
# **Diff-Vorschau zeigt:**
# - Grün: Neue Zeilen
# - Rot: Gelöschte Zeilen
# - Kontext: Umgebende Zeilen
#
# **Aktionen:**
# - ✓ Accept: Änderung übernehmen
# - ✗ Reject: Änderung verwerfen
# - Edit: Manuell anpassen
# - Undo: Rückgängig machen

# %% [markdown]
#
# ## Best Practices für Copilot Edits
#
# **DO:**
# - Klare, spezifische Beschreibungen
# - Kleine, überschaubare Änderungen
# - Jede Datei-Änderung einzeln reviewen
# - Git-Commit vor großen Änderungen
#
# **DON'T:**
# - Zu viele Dateien gleichzeitig
# - Vage Anweisungen
# - Änderungen blind akzeptieren

# %% [markdown]
#
# ## Zusammenfassung
#
# - Copilot Edits: Multi-File-Bearbeitung
# - Edit Mode: Präzise Kontrolle
# - Agent Mode: Autonome Ausführung
# - Diff-Vorschau für alle Änderungen
# - Immer reviewen vor Akzeptieren!
