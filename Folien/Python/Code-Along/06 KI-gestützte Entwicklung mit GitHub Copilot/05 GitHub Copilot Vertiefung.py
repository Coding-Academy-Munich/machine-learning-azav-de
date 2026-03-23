# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>GitHub Copilot Vertiefung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Inline-Vervollständigung: Vertiefung
#
# **In diesem Abschnitt:**
# - Wie Copilot Kontext nutzt
# - Strategien für bessere Vorschläge
# - Best Practices für Produktivität

# %% [markdown]
#
# ## Kontext-Quellen verstehen
#
# **Copilot analysiert (Priorität):**
# 1. Code unmittelbar vor dem Cursor
# 2. Kommentare und Docstrings
# 3. Funktions-/Klassen-Signaturen
# 4. Andere geöffnete Tabs (limitiert)
# 5. Imports und Dependencies
#
# **Tipp:** Öffnen Sie relevante Dateien für besseren Kontext

# %% [markdown]
#
# ## Beispiel: Code vor dem Cursor

# %%
def sort_by_length(words: list[str]) -> list[str]:
    pass


# %%
words = ["apple", "fig", "banana", "kiwi", "cherry"]

# %%

# %% [markdown]
#
# ## Beispiel: Kommentar-basierte Generierung

# %%
# Function to check if a string is a palindrome


# %%

# %%

# %% [markdown]
#
# ## Word-by-Word Akzeptieren
#
# **Tastenkombination:** `Ctrl+→`
#
# **Vorteil:**
# - Nur Teile des Vorschlags übernehmen
# - Mehr Kontrolle über generierten Code
# - Gut bei teilweise korrekten Vorschlägen

# %%
# Write a function split_even_odd() to iterate through a list of strings, and
# return two lists: one with strings of even length, and another with strings of
# odd length.
def split_even_odd(strings: list[str]) -> tuple[list[str], list[str]]:
    pass


# %% [markdown]
#
# ## Zeilenweises Akzeptieren
#
# **Tastenkombination:**
# - Default: keine
# - Anpassen in Settings: `editor.action.inlineSuggest.acceptNextLine`
#
# **Vorteile:**
# - Bei teilweise richtigen Vorschlägen: schneller als Wort-für-Wort
# - Oft weniger aufwendig als Akzeptieren ganzer Blöcke + Korrektur

# %% [markdown]
#
# ## Bessere Vorschläge durch Kontext
#
# ### Funktionssignatur und Docstring
#
# ❌ SCHLECHTER Kontext (vage, ohne Typen):

# %%
def process_data(data):
    pass


# %% [markdown]
#
# ✅ GUTER Kontext (spezifisch, mit Typen):

# %%
import re


def validate_and_sanitize_user_input(
    user_input: str, max_length: int = 100, allowed_chars: str = "a-zA-Z0-9"
) -> str:
    # If Copilot doesn't immediately generate a usable definition, let it
    # generate a docstring before implementing by typing triple quotes and
    # pressing Enter.
    pass


# %% [markdown]
#
# ✅ GUTER Kontext (spezifisch, mit Typen und Docstring):

# %%
def validate_and_sanitize_user_input(
    user_input: str, max_length: int = 100, allowed_chars: str = "a-zA-Z0-9"
) -> str:
    """
    Validate and sanitize user input for security.

    Args:
        user_input: Raw input from user
        max_length: Maximum allowed length
        allowed_chars: Regex pattern for allowed characters

    Returns:
        Sanitized input string

    Raises:
        ValueError: If input is invalid
    """
    # Copilot bekommt jetzt viel besseren Kontext!
    pass


# %% [markdown]
#
# ## Best Practices für Inline-Vorschläge
#
# **DO:**
# - ✅ Aussagekräftige Namen und Type Hints
# - ✅ Docstrings vor Implementierung
# - ✅ Mehrere Vorschläge prüfen (Alt+])
#
# **DON'T:**
# - ❌ Vage Funktionsnamen
# - ❌ Blind akzeptieren
# - ❌ Korrektheit oder Sicherheit vernachlässigen

# %% [markdown]
#
# ## Wann Copilot deaktivieren?
#
# **Situationen:**
# - Beim Lernen neuer Konzepte
# - Bei komplexer Business-Logik
# - Wenn Vorschläge ablenken
# - Bei sensiblen Daten
#
# **Deaktivierung:**
# - Status Bar Icon klicken
# - [Content Exclusion](https://docs.github.com/en/copilot/concepts/context/content-exclusion) für Dateien/Ordner
# - Settings für Dateitypen

# %% [markdown]
#
# ## Next Edit Suggestions (Vorschau)
#
# **Prädiktive Bearbeitung:**
# - Copilot schlägt die *nächste* Änderung vor
# - Pfeile im Gutter zeigen Vorschläge an
# - Basierend auf Ihrem Bearbeitungsmuster
#
# **Wann erscheinen sie:**
# - Nach Akzeptieren einer Vervollständigung
# - Bei repetitiven Änderungen
# - Beim Refactoring ähnlicher Stellen

# %% [markdown]
#
# **Navigation:**
# - `Tab` zum Akzeptieren
# - Pfeiltasten zum Navigieren zwischen Vorschlägen
# - `Esc` zum Ablehnen

# %%
def constant_repeated_many_times(x: int) -> int:
    y1 = x + 2.0
    y2 = y1 + 2.0
    y3 = y2 + 2.0
    y4 = y3 + 2.0
    y5 = y4 + 2.0
    return y5

def constant_also_repeated_many_times(x: int) -> int:
    y1 = x + 3.0
    y2 = y1 + 3.0
    y3 = y2 + 3.0
    y4 = y3 + 3.0
    y5 = y4 + 3.0
    return y5

# %% [markdown]
#
# ## Zusammenfassung
#
# - Kontext ist entscheidend für gute Vorschläge
# - Gute Namen, Kommentare und Docstrings helfen
# - Tastenkombination für Alternativen
# - Wort-/Zeilenweise Vervollständigung
# - Next Edit Suggestions für prädiktive Bearbeitung
