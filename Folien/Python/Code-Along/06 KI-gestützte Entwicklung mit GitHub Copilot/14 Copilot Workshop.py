# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Copilot Workshop</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Copilot Deep Dive Workshop
#
# **Projekt:** CopilotWorkshopStarterKit (BookLibrary)
#
# **Ziele:**
# - Chat Participants praktisch anwenden
# - Copilot Edits für Multi-File-Änderungen
# - Custom Instructions testen
# - Agent Mode für neue Features
#
# **Dauer:** Ca. 45-60 Minuten

# %% [markdown]
#
# ## Vorbereitung
#
# **Öffnen Sie das Projekt:**
# ```bash
# cd examples/CopilotWorkshopStarterKit
# code .
# ```
#
# **Stellen Sie sicher:**
# - Copilot Extension ist aktiv
# - Copilot Chat ist verfügbar
# - Dateien sind geladen

# %% [markdown]
#
# ## Übung 1: @workspace erkunden
#
# **Aufgabe:** Verstehen Sie das Projekt mit @workspace
#
# **Schritte:**
# 1. Öffnen Sie Copilot Chat
# 2. Fragen Sie:
#    ```
#    @workspace Explain the architecture of this library
#    management system. What are the main components?
#    ```
# 3. Fragen Sie:
#    ```
#    @workspace How does the book borrowing process work?
#    ```

# %% [markdown]
#
# ## Übung 2: Copilot Edits nutzen
#
# **Aufgabe:** Input Validierung hinzufügen
#
# **Schritte:**
# 1. Öffnen Sie Copilot Edits (`Ctrl+Shift+I`)
# 2. Fügen Sie `book_api.py` und `member_manager.py` hinzu
# 3. Prompt:
#    ```
#    Add input validation to all methods that accept
#    user input. Validate ISBNs, emails, and dates.
#    Raise ValueError with descriptive messages for invalid input.
#    ```
# 4. Reviewen Sie die Diffs
# 5. Akzeptieren oder passen Sie an

# %% [markdown]
#
# ## Übung 3: Custom Instructions testen
#
# **Aufgabe:** Instructions in Aktion sehen
#
# **Schritte:**
# 1. Öffnen Sie `.github/copilot-instructions.md`
# 2. Lesen Sie die definierten Regeln
# 3. Bitten Sie Copilot:
#    ```
#    Add a function to calculate late fees for overdue
#    books in library.py
#    ```
# 4. Prüfen Sie: Folgt der Code den Instructions?
#    - Type Hints vorhanden?
#    - Docstring vorhanden?
#    - Validierung eingebaut?

# %% [markdown]
#
# ## Übung 4: Agent Mode für neues Feature
#
# **Aufgabe:** Book Reservation System implementieren
#
# **Schritte:**
# 1. Aktivieren Sie Agent Mode in Copilot Edits
# 2. Prompt:
#    ```
#    Implement a book reservation feature:
#    - Add reserve_book(isbn, member_id) method to Library
#    - Books can be reserved if currently borrowed
#    - Add RESERVED status to BookStatus enum
#    - Add tests for reservation functionality
#    ```
# 3. Beobachten Sie wie Agent Dateien findet
# 4. Reviewen Sie alle Änderungen

# %% [markdown]
#
# ## Übung 5: Chat Participants kombinieren
#
# **Aufgabe:** Issue-Vorschlag erstellen
#
# **Schritte:**
# 1. In Copilot Chat:
#    ```
#    @workspace What improvements could be made
#    to the book search functionality?
#    ```
# 2. Basierend auf der Antwort:
#    ```
#    @githubpr Draft an issue description for adding
#    advanced search filters based on your analysis
#    ```

# %% [markdown]
#
# ## Übung 6: Plan Mode nutzen
#
# **Aufgabe:** Feature mit Plan Mode planen und implementieren
#
# **Schritte:**
# 1. Wechseln Sie zum **Plan** Modus im Chat
# 2. Prompt:
#    ```
#    Plan: Add a book recommendation feature that suggests
#    similar books based on genre and author. Consider
#    data models, algorithms, and UI integration.
#    ```
# 3. Reviewen Sie den generierten Plan
# 4. Passen Sie den Plan bei Bedarf an
# 5. Lassen Sie den Plan implementieren

# %% [markdown]
#
# ## Übung 7: Pfadspezifische Instructions
#
# **Aufgabe:** Test-spezifische Instructions erstellen
#
# **Schritte:**
# 1. Erstellen Sie `.github/instructions/tests.instructions.md`:
#    ```markdown
#    ---
#    applyTo: "tests/**/*.py"
#    ---
#    - Use pytest with fixtures from conftest.py
#    - Include edge cases and error conditions
#    - Use parametrized tests where appropriate
#    - Mock external dependencies
#    ```
# 2. Bitten Sie Copilot um Tests für eine Funktion
# 3. Prüfen Sie: Befolgt der Code die Instructions?

# %% [markdown]
#
# ## Bonus: Eigene Instructions erstellen
#
# **Aufgabe:** Instructions für Ihr Projekt anpassen
#
# **Schritte:**
# 1. Erstellen Sie eine neue Instructions-Datei
# 2. Definieren Sie Ihre Coding-Standards:
#    - Naming Conventions
#    - Test-Anforderungen
#    - Dokumentations-Style
# 3. Testen Sie mit neuen Code-Generierungen

# %% [markdown]
#
# ## Dokumentation
#
# **Dokumentieren Sie Ihre Erfahrungen:**
#
# - Welche Übung war am hilfreichsten?
# - Wie gut funktionierten die Custom Instructions?
# - Wie war Agent Mode im Vergleich zu Edit Mode?
# - Welche Chat Participants haben Sie am meisten genutzt?
# - Was würden Sie in Ihrem Alltag einsetzen?

# %% [markdown]
#
# ## Zusammenfassung
#
# **Sie haben gelernt:**
# - @workspace für Codebase-Verständnis
# - Chat-Variablen (#file, #selection, #codebase, #web)
# - Copilot Edits für Multi-File-Änderungen
# - Plan Mode für durchdachte Implementierungen
# - Custom Instructions für projektspezifische Regeln
# - Pfadspezifische Instructions mit applyTo
# - Agent Mode für autonome Implementierung
# - Kombination verschiedener Features
#
# **Nächste Schritte:** Claude Code Deep Dive
