# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Kontext-Variablen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Kontext-Variablen: Präziser Kontext
#
# **Was sind Kontext-Variablen?**
# - Werkzeuge zur präzisen Kontext-Angabe
# - Zugriff mit `#` gefolgt vom Namen
# - Fokussieren Copilot auf relevante Informationen

# %% [markdown]
#
# **Verfügbare Variablen:**
# - `#file` - Bestimmte Datei
# - `#selection` - Markierter Code
# - `#codebase` - Gesamte Codebase
# - `#web` - Web-Suche
# - `#sym` - Spezifisches Symbol

# %% [markdown]
#
# ## `#file`: Datei-Kontext
#
# **Fähigkeiten:**
# - Bezieht spezifische Datei in Kontext ein
# - Ermöglicht gezielte Fragen zu einer Datei
# - Autocomplete zeigt verfügbare Dateien
#
# **Beispiele:**
# ```
# Explain #file:src/weather_api.py focusing on caching
# What does #file:config/settings.json configure?
# Compare #file:old_version.py with #file:new_version.py
# ```

# %% [markdown]
#
# ## `#file`: Praktische Anwendung
#
# **Öffnen Sie `CopilotAdvancedDemo` und fragen Sie:**
#
# ```
# Explain the error handling in #file:src/weather_api.py
# and suggest improvements
# ```
#
# **Copilot wird:**
# - Nur diese Datei analysieren
# - Fokussierte Antwort geben
# - Keine irrelevanten Dateien einbeziehen

# %% [markdown]
#
# ## `#selection`: Markierter Code
#
# **Fähigkeiten:**
# - Bezieht sich auf aktuell markierten Code
# - Ideal für gezielte Refactoring-Fragen
# - Kontext ist exakt das, was Sie sehen
#
# **Beispiele:**
# ```
# Refactor #selection to use list comprehension
# Add type hints to #selection
# What's the time complexity of #selection?
# ```

# %% [markdown]
#
# ## `#codebase`: Gesamte Codebase
#
# **Fähigkeiten:**
# - Durchsucht alle Dateien im Workspace
# - Findet Zusammenhänge projektübergreifend
# - Ähnlich wie @workspace, aber als Variable
#
# **Beispiele:**
# ```
# #codebase Find all usages of the WeatherData class
# #codebase What testing patterns are used in this project?
# #codebase How is authentication implemented?
# ```

# %% [markdown]
#
# ## `#web`: Web-Suche
#
# **Fähigkeiten:**
# - Sucht aktuelle Informationen im Web
# - Ideal für neueste Best Practices
# - Dokumentation und Updates
#
# **Beispiele:**
# ```
# #web What's new in Python 3.12?
# #web Latest pytest best practices for fixtures
# #web How to use the new match statement in Python?
# ```

# %% [markdown]
#
# ## `#sym`: Symbol-Referenz
#
# **Fähigkeiten:**
# - Referenziert spezifische Klassen, Funktionen, Variablen
# - Findet Definition und Verwendung
# - Präziser als Textsuche
#
# **Beispiele:**
# ```
# Add unit tests for #sym:WeatherAPIClient
# Explain the purpose of #sym:WeatherAPIClient.get_forecast
# What methods does #sym:Dashboard have?
# ```

# %% [markdown]
#
# ## Kontext-Variablen kombinieren
#
# **Komplexe Anfragen:**
# ```
# Compare #file:old_api.py with #file:new_api.py
# and explain the differences
# ```
#
# ```
# Refactor #selection using patterns from #file:utils.py
# ```
#
# ```
# #web Find the latest security best practices and
# apply them to #file:auth.py
# ```

# %% [markdown]
#
# ## Participants + Kontext-Variablen
#
# **Maximale Präzision durch Kombination:**
# ```
# @workspace Explain how #sym:WeatherAPIClient
# interacts with #file:data_processor.py
# ```
#
# ```
# @terminal Run tests for #file:test_weather_api.py
# ```
#
# ```
# @githubpr Create an issue for the bug in #selection
# ```

# %% [markdown]
#
# ## Best Practices: Kontext-Variablen
#
# **Wann welche Variable nutzen:**
# - `#file`: Gezielte Fragen zu einer Datei
# - `#selection`: Refactoring markierter Code
# - `#codebase`: Projektweite Suche
# - `#web`: Aktuelle externe Informationen
# - `#sym`: Präzise Symbol-Referenz
#
# **Tipp:** Je präziser der Kontext, desto besser die Antwort

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Participants (`@`)**: Spezialisierte Agenten
#   - `@workspace`, `@githubpr`, `@terminal`, `@vscode`
# - **Kontext-Variablen (`#`)**: Präziser Kontext
#   - `#file`, `#selection`, `#codebase`, `#web`, `#sym`
# - Kombination für maximale Effektivität
# - Präziser Kontext = bessere Antworten
