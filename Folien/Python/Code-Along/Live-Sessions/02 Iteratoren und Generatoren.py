# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Iteratoren und Generatoren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Warum Iteratoren?
#
# - Sie haben `for`-Schleifen schon oft benutzt
# - Aber was passiert eigentlich hinter den Kulissen?
# - Das Verständnis von Iteratoren ermöglicht:
#   - Speichereffiziente Verarbeitung großer Datenmengen
#   - Verarbeitung von Datenströmen (z.B. Log-Dateien, Sensordaten)
#   - Eigene iterierbare Typen zu erstellen

# %% [markdown]
#
# # Iteration über verschiedene Typen
#
# Die `for`-Schleife in Python funktioniert mit vielen verschiedenen Typen:

# %%

# %%

# %% [markdown]
#
# Wie schafft Python das? Der Mechanismus dahinter heißt **Iterator**.

# %% [markdown]
#
# # Was ist ein Iterator?
#
# - Ein **Iterator** ist ein Objekt, das Elemente **nacheinander** liefert
# - Die Funktion `iter()` erzeugt einen Iterator aus einem iterierbaren Objekt
# - Die Funktion `next()` holt das **nächste Element** vom Iterator

# %%
my_list = [1, 2, 3]

# %% [markdown]
#
# Mit `iter()` erzeugen wir einen Iterator:

# %%

# %% [markdown]
#
# Mit `next()` holen wir das erste Element:

# %%

# %% [markdown]
#
# Jeder weitere Aufruf von `next()` liefert das nächste Element.
# Der Iterator **merkt sich** seine aktuelle Position:

# %%

# %%

# %% [markdown]
#
# Was passiert, wenn wir `next()` noch einmal aufrufen?
# Der Iterator ist **erschöpft** - es gibt keine Elemente mehr.
# Python wirft dann eine `StopIteration` Exception:

# %%

# %% [markdown]
#
# # Die `for`-Schleife und Iteratoren
#
# Die `for`-Schleife verwendet Iteratoren intern.
# Diese beiden Schleifen machen das gleiche:

# %%

# %%

# %% [markdown]
#
# # Was macht ein Objekt iterierbar?
#
# - Ein Objekt ist **iterierbar** wenn es eine `__iter__` Methode hat
# - Die `__iter__` Methode muss einen Iterator zurückgeben
# - Der Typ `Iterable` aus `typing` kann verwendet werden um das zu prüfen

# %%

# %%

# %%

# %% [markdown]
#
# # Eigene iterierbare Klassen
#
# Wir können unsere eigenen Klassen iterierbar machen.
# Dazu implementieren wir die `__iter__` Methode:

# %%

# %%

# %%

# %%

# %% [markdown]
#
# # Wichtig: Iteratoren sind einmalig verwendbar!
#
# - Nachdem alle Elemente abgerufen wurden ist der Iterator **erschöpft**
# - Ein erschöpfter Iterator liefert keine weiteren Elemente
# - Um erneut zu iterieren brauchen wir einen **neuen** Iterator
# - Das ist wie ein Buch lesen: Am Ende angekommen müssen wir von vorne beginnen

# %%
list_it = iter(my_list)
obj_it = iter(my_obj)

# %% [markdown]
#
# Der erste Aufruf von `tuple()` verbraucht alle Elemente:

# %%

# %% [markdown]
#
# Der zweite Aufruf liefert ein leeres Tupel - der Iterator ist erschöpft:

# %%

# %% [markdown]
#
# Das gleiche gilt für jeden Iterator:

# %%

# %%

# %% [markdown]
#
# Iteratoren können auch zum Entpacken verwendet werden:

# %%

# %% [markdown]
#
# # Verschiedene Iterator-Arten
#
# Manche Typen bieten mehrere Möglichkeiten zur Iteration.
# Dictionaries haben z.B. separate Iteratoren für Schlüssel, Werte und Paare:

# %%

# %% [markdown]
#
# Standard-Iteration über ein Dictionary liefert die **Schlüssel**:

# %%

# %%

# %% [markdown]
#
# Mit `.values()` iterieren wir über die **Werte**:

# %%

# %% [markdown]
#
# Mit `.items()` bekommen wir **Schlüssel-Wert-Paare**:

# %%

# %%

# %% [markdown]
#
# ## Wie funktioniert die `for`-Schleife intern?
#
# Die `for`-Schleife ist eigentlich eine Abkürzung für eine `while`-Schleife
# mit Iterator:

# %%

# %% [markdown]
#
# Diese `for`-Schleife:

# %%

# %% [markdown]
#
# Entspricht diesem Code:

# %%

# %% [markdown]
#
# Die `for`-Schleife:
# 1. Ruft `iter()` auf um einen Iterator zu bekommen
# 2. Ruft wiederholt `next()` auf
# 3. Fängt `StopIteration` ab um die Schleife zu beenden

# %% [markdown]
#
# # Generatoren
#
# ## Das Problem mit Listen
#
# - Listen speichern **alle Elemente** gleichzeitig im Speicher
# - Bei großen Datenmengen kann das problematisch sein
# - Beispiel: Eine Liste mit 10 Millionen Zahlen braucht viel Speicher

# %% [markdown]
#
# ## Die Lösung: Generator Expressions
#
# - Ein **Generator** erzeugt Elemente **bei Bedarf** - eines nach dem anderen
# - Er speichert nur die **Anleitung** wie Elemente erzeugt werden
# - Wie ein Rezept vs. ein fertiges Gericht:
#   - Liste = fertiges Gericht (alle Portionen sind schon da)
#   - Generator = Rezept (Portionen werden erst bei Bedarf zubereitet)

# %% [markdown]
#
# ## Syntax: List Comprehension vs. Generator Expression
#
# - List Comprehension: `[ausdruck for x in iterable]` - eckige Klammern
# - Generator Expression: `(ausdruck for x in iterable)` - runde Klammern

# %% [markdown]
#
# Eine List Comprehension erzeugt sofort alle Werte:

# %%
# !pip install icecream

# %%
from icecream import ic

# %%

# %%

# %%

# %% [markdown]
#
# Eine Generator Expression erzeugt ein Generator-Objekt:

# %%

# %%

# %% [markdown]
#
# Der Generator berechnet die Werte erst beim Iterieren:

# %%

# %% [markdown]
#
# ## Generatoren sind Iteratoren - also einmalig verwendbar!
#
# Nach dem ersten Durchlauf ist der Generator erschöpft:

# %%

# %% [markdown]
#
# Es wird nichts ausgegeben, weil der Generator bereits erschöpft ist.
# Um erneut zu iterieren, müssen wir einen **neuen** Generator erstellen.

# %% [markdown]
#
# ## Komplexere Generator Expressions
#
# Generator Expressions können Filter und verschachtelte Schleifen enthalten:

# %%

# %% [markdown]
#
# ## Generatoren und `iter()`
#
# Ein Generator ist sein eigener Iterator:

# %%

# %%

# %% [markdown]
#
# `iter(gen)` gibt den gleichen Generator zurück.
# Wir können `next()` direkt auf dem Generator aufrufen:

# %%

# %%

# %%

# %% [markdown]
#
# Nach dem letzten Element ist der Iterator erschöpft:

# %%

# %%

# %% [markdown]
#
# ## Workshop: Generator-Expressions
#
# Bearbeiten Sie die folgenden Aufgaben:
#
# 1. Berechnen Sie die Summe der ersten 100 Quadratzahlen (0², 1², 2², ... 99²)
#    unter Verwendung einer Generator-Expression.
#
# 2. Berechnen Sie die Summe aller Zahlen zwischen 100 und 500 (einschließlich),
#    die durch 7 teilbar sind, unter Verwendung einer Generator-Expression.
#
# 3. Schreiben Sie eine Funktion
#    `all_powers(numbers: Iterable[int], powers: Iterable[int])`,
#    die eine Liste zurückgibt, deren Elemente Tupel sind.
#    Jedes Tupel enthält alle Potenzen aus `powers` für ein Element aus `numbers`.
#
#    Beispiel: `all_powers([2, 3], [0, 1, 2])` ergibt
#    `[(1, 2, 4), (1, 3, 9)]` (also 2⁰, 2¹, 2² und 3⁰, 3¹, 3²)


# %%

# %%

# %%

# %%
assert all_powers(range(3), range(3)) == [(1, 0, 0), (1, 1, 1), (1, 2, 4)]

# %%
assert all_powers([10, 11, 12], [2, 3]) == [(100, 1000), (121, 1331), (144, 1728)]
