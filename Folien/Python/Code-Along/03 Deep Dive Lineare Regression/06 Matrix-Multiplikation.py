# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Matrix-Multiplikation</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Vektoren in NumPy
#
# Bisher haben wir mit Vektoren gearbeitet, aber was sind sie wirklich?
#
# - 1D-Arrays in NumPy
# - Können als Zeilen- oder Spaltenvektoren verwendet werden
# - NumPy behandelt sie flexibel

# %%
import numpy as np
from display_utils import side_by_side, with_shape

# %% [markdown]
#
# ## Zeilen- und Spaltenvektoren
#
# In der Mathematik unterscheiden wir:
#
# **Spaltenvektor** (column vector):
# $$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$
#
# **Zeilenvektor** (row vector):
# $$\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$$

# %% [markdown]
#
# ### NumPy's 1D-Arrays
#
# Ein einfacher Vektor in NumPy:

# %%

# %%

# %%

# %%

# %% [markdown]
#
# **Wichtig:** Dies ist weder Zeilen- noch Spaltenvektor!
# - Es ist ein 1D-Array
# - Shape ist `(3,)` nicht `(3, 1)` oder `(1, 3)`
# - NumPy passt es automatisch an den Kontext an

# %% [markdown]
#
# ## Vektoren als 2D-Matrizen
#
# Wir können explizit Zeilen- und Spaltenvektoren erstellen:

# %% [markdown]
#
# ### Spaltenvektor (3×1 Matrix)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ### Zeilenvektor (1×3 Matrix)

# %%

# %%

# %% [markdown]
#
# ## `reshape()`: Vektoren umformen
#
# Mit `reshape()` können wir die Form eines Arrays ändern:

# %% [markdown]
#
# Von 1D zu Spaltenvektor:

# %%
v = np.array([1, 2, 3])

# %%

# %%

# %%

# %% [markdown]
#
# Von 1D zu Zeilenvektor:

# %%

# %%

# %% [markdown]
#
# ### `reshape()` mit -1
#
# NumPy kann eine Dimension automatisch berechnen:

# %%
v = np.array([1, 2, 3, 4, 5, 6])

# %%

# %% [markdown]
#
# - -1 bedeutet "berechne diese Dimension"
# - Das ist nur für eine Dimension erlaubt

# %%

# %%

# %%

# %%


# %% [markdown]
#
# Zeilen- und Spaltenvektor mit `-1`:

# %%
v = np.array([1, 2, 3])

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Matrix × Vektor: Die Details
#
# Schauen wir genauer hin, was bei Matrix × Vektor passiert:

# %%
A = np.array([[1, 2, 3], [4, 5, 6]])

# %%
# %with_shape A

# %%
v = np.array([1, 1, 2])

# %%
# %with_shape v

# %%

# %% [markdown]
#
# Was passiert, wenn wir einen expliziten Spaltenvektor verwenden?

# %%
v_col = v.reshape(-1, 1)

# %%
# %with_shape A, v_col

# %%


# %% [markdown]
#
# Was passiert, wenn wir einen expliziten Zeilenvektor verwenden?

# %%
v_row = v.reshape(1, -1)

# %%
# %with_shape A, v_row

# %%

# %% [markdown]
#
# Beachte:
# - Mit 1D-Vektor: Ergebnis ist 1D `(2,)`
# - Mit Spaltenvektor: Ergebnis ist 2D `(2, 1)`
#   - Gleiche Werte, unterschiedliche Formen!
# - Mit Zeilenvektor: Fehler (Dimensionen passen nicht)

# %% [markdown]
#
# ## Vektor × Matrix
#
# Was passiert, wenn wir den Vektor auf die linke Seite setzen?

# %%
A = np.array([[1, 2, 3], [4, 5, 6]])
v = np.array([1, 2, 3])

# %%
# %with_shape v, A


# %% [markdown]
#
# ### Mit Zeilenvektor

# %%

# %%

# %% [markdown]
#
# ### Mit Spaltenvektor

# %%

# %%

# %% [markdown]
#
# Beachte:
# - Mit 1-D Vektor: Ergebnis ist 1D `(3,)`
# - Mit Zeilenvektor: Ergebnis is 2D `(1, 3)`
#   - Gleiche Werte, unterschiedliche Formen!
# - Mit Spaltenvektor: Fehler (Dimensionen passen nicht)

# %% [markdown]
#
# ## Transponieren: Von Zeile zu Spalte
#
# Die Transponierte `.T` tauscht Zeilen und Spalten:

# %%
v_row = np.array([[1, 2, 3]])

# %%

# %%

# %%


# %% [markdown]
#
# **Achtung:** Transponieren von 1D-Arrays hat keine Wirkung!

# %%
v = np.array([1, 2, 3])

# %%
# %with_shape v

# %%

# %% [markdown]
#
# Lösung: Erst zu 2D umformen, dann transponieren:

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Matrix × Matrix: Die Grundidee
#
# Jetzt wo wir Zeilen- und Spaltenvektoren verstehen...
# - Bisher: Eine Matrix ist eine Sammlung von Zeilenvektoren
# - **Eine Matrix ist auch eine Sammlung von Spaltenvektoren!**

# %% [markdown]
#
# - Matrix × Matrix = $\begin{bmatrix}
#     ⸺ & ZV_0 & ⸺\\
#     ⸺ & \dots & ⸺\\
#     ⸺ & ZV_{m-1} & ⸺
#   \end{bmatrix} × \begin{bmatrix}
#     ︱ & ︱ & ︱\\
#     SV_0 & \dots & SV_{p-1}\\
#     ︱ & ︱ & ︱
#   \end{bmatrix}$
# - Multipliziere jeden Zeilenvektor mit jedem Spaltenvektor
# - Ergebnisse werden zu Einträgen der Ergebnismatrix
#   - ZVᵢ × SVⱼ → eᵢⱼ
#   - Zeilenvektor × Matrix → Zeilenvektor des Ergebnisses
#   - Matrix × Spaltenvektor → Spaltenvektor des Ergebnisses

# %% [markdown]
#
# Beispiel:

# %%
A = np.array([[1, 2, 3], [4, 5, 6]])

# %%
# %with_shape A

# %%
B = np.array([[10, 11], [20, 22], [30, 33]])

# %%

# %% [markdown]
#
# ### Spaltenweise Betrachtung
#
# - Erste Spalte von B: SV₁

# %%
b1 = B[:, 0].reshape(-1, 1)

# %%

# %%

# %% [markdown]
#
# - Zweite Spalte von B: SV₂

# %%

# %%

# %%

# %% [markdown]
#
# Gesamtergebnis: A × B

# %%
# %with_shape A, B

# %%
# %with_shape A @ b1

# %%
# %with_shape A @ b2

# %%

# %% [markdown]
#
# ## Dimensionen bei Matrix-Multiplikation
#
# Für $A × B$:
# - $A$ hat Form $(m, n)$
# - $B$ hat Form $(n, p)$
# - Ergebnis hat Form $(m, p)$
#
# **Wichtig:** Die inneren Dimensionen müssen übereinstimmen!
# - Anzahl Spalten von $A$ = Anzahl Zeilen von $B$

# %% [markdown]
#
# Beispiele für gültige Multiplikationen:

# %%
A1 = np.random.rand(2, 3)
B1 = np.random.rand(3, 4)

# %%
C1 = A1 @ B1

# %%
f"{A1.shape} × {B1.shape} = {C1.shape}"

# %%
A2 = np.random.rand(5, 2)
B2 = np.random.rand(2, 7)

# %%
C2 = A2 @ B2

# %%
f"{A2.shape} × {B2.shape} = {C2.shape}"

# %% [markdown]
#
# Ungültige Multiplikation:

# %%
A_bad = np.random.rand(2, 3)
B_bad = np.random.rand(4, 5)

# %%
try:
    C_bad = A_bad @ B_bad
except ValueError as e:
    print(f"Error: Cannot multiply (2, 3) × (4, 5)")
    print(f"Inner dimensions 3 and 4 don't match!")

# %% [markdown]
#
# ## Visualisierung: Dimensionen verstehen

# %%
A_vis = np.arange(6).reshape(2, 3)
B_vis = np.arange(12).reshape(3, 4)

# %%
# %with_shape A_vis, B_vis

# %%
C_vis = A_vis @ B_vis

# %%

# %% [markdown]
#
# <img src="img/matmul.png"
#      style="width: 90%; margin-left: auto; margin-right: auto; display: block;">

# %%

# %% [markdown]
#
# ## Matrix-Multiplikation ist nicht kommutativ
#
# **Wichtig:** $A × B \neq B × A$
#
# Die Reihenfolge ist wichtig!

# %% [markdown]
#
# Beispiel:

# %%
A = np.array([[1, 2, 4], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

# %%
# %side_by_side A, B

# %%

# %%

# %% [markdown]
#
# Beispiel:

# %%
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# %%
# %side_by_side A, B

# %%

# %%


# %% [markdown]
#
# ## Zusammenfassung
#
# **Vektoren in NumPy:**
# - 1D-Arrays sind weder Zeilen- noch Spaltenvektoren
# - NumPy passt sie flexibel an
# - Für explizite Kontrolle: `reshape()` verwenden
#
# **`reshape()`:**
# - Ändert die Form eines Arrays
# - `-1` lässt NumPy eine Dimension berechnen
# - Wichtig für Vektoren: `reshape(-1, 1)` für Spalten

# %% [markdown]
#
# **Matrix × Matrix:**
# - Verallgemeinerung von Matrix × Vektor
# - Dimensionen: $(m, n) × (n, p) = (m, p)$
# - Nicht kommutativ: $A × B \neq B × A$

# %% [markdown]
#
# ## Workshop: Vektoren und Matrizen

# %% [markdown]
#
# ### Aufgabe 1: Vektoren umformen
#
# Gegeben ist der Vektor:
# ```python
# v = np.array([1, 2, 3, 4])
# ```
#
# 1. Formen Sie ihn in einen Spaltenvektor (4×1) um
# 2. Formen Sie ihn in einen Zeilenvektor (1×4) um
# 3. Formen Sie ihn in eine 2×2 Matrix um

# %% [markdown]
#
# ### Aufgabe 2: Matrix-Multiplikation üben
#
# Gegeben sind:
# ```python
# A = np.array([[1, 2, 3],
#               [4, 5, 6]])
# B = np.array([[1, 4],
#               [2, 5],
#               [3, 6]])
# ```
#
# 1. Berechnen Sie $A × B$
# 2. Versuchen Sie $B × A$ zu berechnen - funktioniert das?
# 3. Was ist der Unterschied?

# %% [markdown]
#
# ### Aufgabe 3: Spaltenweise Berechnung
#
# Gegeben ist:
# ```python
# A = np.array([[2, 1],
#               [3, 4]])
# B = np.array([[1, 0, 2],
#               [0, 1, 1]])
# ```
#
# Berechnen Sie $A × B$ auf zwei Arten:
#
# 1. Direkt mit `@`
# 2. Spaltenweise (multipliziere A mit jeder Spalte von B einzeln)
#
# Vergleichen Sie die Ergebnisse!
#
# **Hinweis:** Verwenden Sie `np.column_stack()`, um Spalten zu kombinieren,
# falls nötig.

# %%

# %% [markdown]
#
# ### Aufgabe 4: Transpose und Reshape
#
# Gegeben ist:
# ```python
# v = np.array([1, 2, 3, 4, 5, 6])
# ```
#
# 1. Formen Sie `v` in eine 2×3 Matrix um
# 2. Transponieren Sie das Ergebnis
# 3. Formen Sie das Ergebnis wieder in einen 1D-Vektor um

# %%
