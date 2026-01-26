# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Generator-Funktionen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Warum Generator-Funktionen?
#
# Generator Expressions sind praktisch, aber sie haben Grenzen:
#
# - Was, wenn wir **unendlich viele** Werte erzeugen wollen?
# - Was, wenn wir **komplexe Logik** brauchen (Schleifen, Bedingungen)?
# - Was, wenn wir **externen Zustand** verarbeiten wollen?
#
# Für diese Fälle gibt es **Generator-Funktionen**.

# %% [markdown]
#
# ## Beispiele für Anwendungsfälle
#
# - Alle natürlichen Zahlen erzeugen (unendliche Sequenz)
# - Zeilen aus einer Datei lesen und transformieren
# - Nur die ersten N Elemente eines Iterators nehmen
# - Elemente filtern oder zusammenfassen

# %% [markdown]
#
# # Das `yield` Schlüsselwort
#
# - `yield` ist wie ein **Pause-Button** für eine Funktion
# - Wenn Python auf `yield` trifft:
#   1. Der aktuelle Wert wird zurückgegeben
#   2. Der Funktionszustand wird **gespeichert**
#   3. Die Funktion **pausiert**
# - Beim nächsten `next()` Aufruf wird die Funktion **fortgesetzt**

# %% [markdown]
#
# ## `yield` vs. `return`
#
# | `return` | `yield` |
# |----------|---------|
# | Beendet die Funktion | Pausiert die Funktion |
# | Gibt einen Wert zurück | Gibt einen Wert zurück |
# | Zustand wird verworfen | Zustand wird gespeichert |
# | Kann nur einmal aufgerufen werden | Kann mehrmals fortgesetzt werden |

# %% [markdown]
#
# ## Beispiel: Unendliche Zahlenfolge
#
# Diese Funktion erzeugt **alle** ganzen Zahlen ab einem Startwert:


# %%

# %% [markdown]
#
# Der Aufruf der Funktion erzeugt einen **Generator** (keinen Wert!):

# %%

# %% [markdown]
#
# ## Schritt für Schritt durch den Generator
#
# Schauen wir uns an, was bei jedem `next()` passiert:

# %%
gen = integers()

# %% [markdown]
#
# **Erster `next()` Aufruf:**
# - `n = 0` wird initialisiert
# - `yield 0` gibt 0 zurück und pausiert

# %%

# %% [markdown]
#
# **Zweiter `next()` Aufruf:**
# - Fortsetzung nach `yield`: `n += 1` → `n = 1`
# - `yield 1` gibt 1 zurück und pausiert

# %%

# %% [markdown]
#
# Und so weiter - **für immer**, da die `while True` Schleife nie endet:

# %%

# %%

# %% [markdown]
#
# ## Generator in einer `for`-Schleife
#
# Da der Generator unendlich ist, müssen wir manuell abbrechen:

# %%

# %% [markdown]
#
# ## Mini-Workshop: Einfache Generator-Funktion
#
# Schreiben Sie eine Generatorfunktion `one_based_range(n)`, die wie `range(n)`
# funktioniert, aber von **1 bis einschließlich n** iteriert (statt von 0 bis n-1).
#
# Beispiel: `list(one_based_range(4))` soll `[1, 2, 3, 4]` ergeben.

# %%

# %%
assert list(one_based_range(4)) == [1, 2, 3, 4]


# %% [markdown]
#
# ## Mini-Workshop: Flexiblere Range-Funktion
#
# Schreiben Sie eine Generatorfunktion `inclusive_range()`, die ähnlich wie `range()`
# funktioniert, aber:
# - Die **obere Grenze einschließt**
# - Mit **1 beginnt** wenn nur ein Argument gegeben wird
#
# Die Funktion soll mit 1, 2 oder 3 Argumenten funktionieren:
# - `inclusive_range(end)` → 1 bis end
# - `inclusive_range(begin, end)` → begin bis end
# - `inclusive_range(begin, end, step)` → begin bis end mit Schrittweite

# %%

# %%

# %%
assert list(inclusive_range(3)) == [1, 2, 3]

# %%
assert list(inclusive_range(2, 4)) == [2, 3, 4]

# %%
assert list(inclusive_range(2, 2)) == [2]

# %%
assert list(inclusive_range(2, 1)) == []

# %%
assert list(inclusive_range(2, 6, 2)) == [2, 4, 6]


# %% [markdown]
#
# # Generatoren zur Iterator-Verarbeitung
#
# Generator-Funktionen sind ideal um Iteratoren zu **transformieren**.
# Sie sind die Bausteine für **Daten-Pipelines**.

# %% [markdown]
#
# ## Die `take()` Funktion
#
# Nimmt nur die ersten `n` Elemente eines Iterators:

# %%

# %% [markdown]
#
# Damit können wir endlich viele Elemente aus einem unendlichen Generator holen:

# %%


# %% [markdown]
#
# ## Die `drop()` Funktion
#
# Entfernt die ersten `n` Elemente eines Iterators.
#
# **Beachte:** Dies ist **keine** Generator-Funktion! Sie gibt den
# modifizierten Iterator direkt zurück:

# %%

# %% [markdown]
#
# Kombinieren wir `take` und `drop`:

# %%

# %% [markdown]
#
# Das ergibt die Zahlen 2, 3, 4 (wir überspringen 0 und 1, dann nehmen wir 3 Zahlen).

# %% [markdown]
#
# # Das `yield from` Statement
#
# Manchmal wollen wir alle Elemente eines anderen Iterators zurückgeben.
# Statt einer Schleife können wir `yield from` verwenden:

# %% [markdown]
#
# ## Mit Schleife vs. mit `yield from`
#
# Diese beiden Funktionen machen das gleiche:

# %%

# %%


# %% [markdown]
#
# Mit `yield from` ist es kürzer:

# %%

# %%

# %% [markdown]
#
# `yield from it` ist äquivalent zu:
# ```python
# for element in it:
#     yield element
# ```

# %% [markdown]
#
# # Rekursive Generatoren
#
# `yield from` ist besonders nützlich für **rekursive** Generatoren.
# Damit können wir verschachtelte Strukturen durchlaufen.

# %% [markdown]
#
# ## Beispiel: Verschachtelte Listen abflachen
#
# Gegeben eine verschachtelte Liste wie `[1, [2, 3], [4, [5, 6]]]`,
# wollen wir alle Elemente nacheinander ausgeben: `1, 2, 3, 4, 5, 6`

# %%

# %%

# %% [markdown]
#
# **Wie funktioniert das?**
# - Für jedes Element prüfen wir: Ist es eine Liste?
# - Falls ja: Rekursiv `flatten` aufrufen und alle Ergebnisse durchreichen
# - Falls nein: Das Element direkt zurückgeben

# %% [markdown]
#
# ## Mini-Workshop: Binärbaum durchlaufen
#
# Ein Binärbaum ist eine rekursive Datenstruktur.
# Schreiben Sie einen Generator `traverse(tree)`, der alle Werte
# eines Binärbaums in der Reihenfolge Links → Wurzel → Rechts zurückgibt.

# %%
from dataclasses import dataclass
from typing import Optional


# %%
@dataclass
class Bintree:
    value: object = None
    left: Optional["Bintree"] = None
    right: Optional["Bintree"] = None


# %% [markdown]
#
# **Hinweis:** Verwenden Sie `yield from` für die rekursiven Aufrufe.
# Prüfen Sie, ob `tree` (bzw. `tree.left`/`tree.right`) existiert,
# bevor Sie darauf zugreifen.

# %%

# %%
tree1 = Bintree(1)
tree2 = Bintree(2, Bintree(1), Bintree(3))
tree3 = Bintree(
    1, Bintree(0), Bintree(3, Bintree(2), Bintree(5, Bintree(4), Bintree(6)))
)

# %%
assert tuple(traverse(tree1)) == (1,)

# %%
assert tuple(traverse(tree2)) == (1, 2, 3)

# %%
assert tuple(traverse(tree3)) == tuple(range(7))
