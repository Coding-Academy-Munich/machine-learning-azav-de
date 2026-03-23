# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Testing-Strategien</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Test-Driven Development mit KI
#
# - **Tests zuerst schreiben**, dann implementieren lassen
# - Bestätigen, dass Tests fehlschlagen
# - KI implementiert gegen die Tests
# - **Vorteil:** Klare Zielvorgabe für die KI

# %% [markdown]
#
# ## TDD-Workflow mit KI
#
# ```
# 1. User: Schreibe Tests für eine Funktion calculate_discount()
#          die 10% Rabatt für Bestellungen über 100€ gibt
#
# 2. Claude: [Erstellt Tests]
#
# 3. User: Führe die Tests aus
# ```
#
# ```
# 4. Claude: 3 Tests fehlgeschlagen (Funktion existiert nicht)
#
# 5. User: Implementiere die Funktion
#
# 6. Claude: [Implementiert] - Alle Tests bestanden!
# ```

# %% [markdown]
#
# ## KI-Änderungen an Tests überwachen
#
# **Kritisch:** KI kann fehlschlagende Tests "reparieren" indem sie den Test
# ändert, nicht den Code!
#
# - **Immer** Test-Diffs separat von Code-Diffs reviewen
# - Niemals Test-Modifikationen blind akzeptieren
# - **Fragen Sie sich:**
#   - Warum hat die KI den Test geändert?
#   - Ist die ursprüngliche Erwartung falsch?
#   - Oder hat die KI den einfachen Weg gewählt?

# %% [markdown]
#
# ## Beispiel: Gefährliche Test-Änderung
#
# **Original-Test:**
# ```python
# def test_divide_by_zero():
#     with pytest.raises(ValueError):
#         calculator.divide(10, 0)
# ```
#
# **KI "repariert" so:**
# ```python
# def test_divide_by_zero():
#     result = calculator.divide(10, 0)
#     assert result == float('inf')  # Geänderte Erwartung!
# ```
#
# Der Test besteht jetzt, aber das Verhalten ist falsch!

# %% [markdown]
#
# ## Aussagekräftige Assertions sicherstellen
#
# - KI-generierte Tests auf Qualität prüfen
# - **Warnsignale:**
#   - Leere Assertions (`assert True`)
#   - Triviale Tests die immer bestehen
#   - Fehlende Edge Cases
#   - Nur Happy-Path getestet
# - Coverage- und Mutation-Testing verwenden

# %% [markdown]
#
# ## Beispiele für schwache vs. starke Tests
#
# **Schwach:**
# ```python
# def test_calculate_total():
#     result = calculate_total([10, 20, 30])
#     assert result is not None  # Testet fast nichts
# ```
#
# **Stark:**
# ```python
# def test_calculate_total():
#     assert calculate_total([10, 20, 30]) == 60
#     assert calculate_total([]) == 0
#     assert calculate_total([-5, 5]) == 0
# ```

# %% [markdown]
#
# ## Test-Suite als Sicherheitsnetz
#
# - **Schnelle Tests** = schnelles Feedback bei KI-Fehlern
# - Tests häufig während KI-Sessions ausführen
# - Test-Fehler als "Lernmomente" für Prompt-Verfeinerung nutzen
# - **Regel:** Keine KI-Änderung akzeptieren die Tests bricht

# %% [markdown]
#
# ## Zusammenfassung: Testing-Strategien
#
# | Strategie | Umsetzung |
# |-----------|-----------|
# | TDD mit KI | Tests zuerst, dann implementieren lassen |
# | Test-Änderungen überwachen | Diffs separat reviewen |
# | Starke Assertions | Konkrete Erwartungen, Edge Cases |
# | Schnelles Feedback | Tests häufig ausführen |

# %% [markdown]
#
# ## Workshop: Test-Qualitäts-Review
#
# **Teil 1: Schwache Tests identifizieren**
#
# Analysieren Sie folgende KI-generierte Tests und identifizieren Sie Probleme:
#
# ```python
# def test_user_creation():
#     user = User("Alice", "alice@example.com")
#     assert user is not None
#
# def test_email_validation():
#     assert validate_email("test@test.com")
# ```
#
# ```python
# def test_password_strength():
#     result = check_password("abc")
#     # TODO: add assertions
# ```

# %% [markdown]
#
# ## Workshop: Test-Qualitäts-Review (Fortsetzung)
#
# **Teil 2: Geänderte Tests erkennen**
#
# Die KI hat folgende Änderung vorgeschlagen. Was ist das Problem?
#
# ```diff
# - def test_max_items_limit():
# -     cart = ShoppingCart()
# -     for i in range(101):
# -         cart.add_item(f"item_{i}")
# -     assert len(cart.items) == 100  # Max 100 items
# + def test_max_items_limit():
# +     cart = ShoppingCart()
# +     for i in range(101):
# +         cart.add_item(f"item_{i}")
# +     assert len(cart.items) == 101  # All items added
# ```

# %% [markdown]
#
# ## Workshop: Test-Qualitäts-Review (Fortsetzung)
#
# **Teil 3: Praktische Übung**
#
# 1. Bitten Sie die KI, Tests für eine einfache Funktion zu generieren
# 2. Reviewen Sie die Tests kritisch:
#    - Sind die Assertions aussagekräftig?
#    - Werden Edge Cases getestet?
#    - Fehlen wichtige Szenarien?
# 3. Bitten Sie die KI, die Tests zu verbessern
# 4. Vergleichen Sie vorher/nachher
