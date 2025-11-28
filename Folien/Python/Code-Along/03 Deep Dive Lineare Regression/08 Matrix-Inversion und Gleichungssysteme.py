# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Matrix-Inversion und Gleichungssysteme</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Motivation: Gleichungen lösen
#
# **Bei normalen Zahlen:**
# - Gegeben: $3 × x = 12$
# - Lösung: $x = 12 / 3 = 4$
# - Wir dividieren durch 3
#
# **Bei Matrizen:**
# - Gegeben: $A × x = b$
# - Problem: Man kann nicht durch Matrizen dividieren!
# - Lösung: ???

# %% [markdown]
#
# Beispiel eines Gleichungssystems:
#
# $$
# \begin{align*}
# 3x_1 + 2x_2 &= 7 \\
# 4x_1 + x_2 &= 6
# \end{align*}
# $$
#
# Als Matrix-Gleichung: $A × x = b$ mit
#
# $$
# A = \begin{bmatrix} 3 & 2 \\ 4 & 1 \end{bmatrix}, \quad
# x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \quad
# b = \begin{bmatrix} 7 \\ 6 \end{bmatrix}
# $$

# %%
import numpy as np
import matplotlib.pyplot as plt
from linear_regression_plots import plot_matrix_inverse_transformation
from display_utils import side_by_side

# %%
A = np.array([[3, 2], [4, 1]])

# %%
b = np.array([7, 6])

# %%


# %% [markdown]
#
# ## Die Identitätsmatrix
#
# Um "Division" für Matrizen zu verstehen, brauchen wir erst ein Konzept...

# %% [markdown]
#
# ### Was ist die Identitätsmatrix?
#
# - Quadratische Matrix (gleich viele Zeilen wie Spalten)
# - Einsen auf der Diagonale
# - Nullen überall sonst
# - Symbol: $I$ oder $I_n$ (für $n×n$ Matrix)
#
# **Verhält sich wie die Zahl 1!**

# %% [markdown]
#
# Beispiele:
#
# 2×2 Identitätsmatrix:
#
# $$I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

# %%

# %%

# %% [markdown]
#
# 3x3 Identitätsmatrix
#
# $$I_3 = \begin{bmatrix}1 & 0 & 0\\0 & 1 & 0\\0 & 0 & 1\end{bmatrix}$$

# %%

# %%

# %% [markdown]
#
# 4x4 Identitätsmatrix
#
# $$I_4 = \begin{bmatrix}1 & 0 & 0 & 0\\0 & 1 & 0 & 0\\0 & 0 & 1 & 0\\0 & 0 & 0 & 1\end{bmatrix}$$

# %%

# %%

# %% [markdown]
#
# ## Eigenschaft der Identitätsmatrix
#
# Die Identitätsmatrix verhält sich wie die **1** bei der Multiplikation:
#
# $$A × I = I × A = A$$
#
# Sie ändert Matrizen nicht!

# %% [markdown]
#
# ### Beispiel

# %%
A = np.array([[2.0, 3.0], [4.0, 5.0]])

# %%

# %%

# %% [markdown]
#
# Mit größeren Matrizen:

# %%
B = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# %%

# %%

# %% [markdown]
#
# ## Matrix-Inversion: Die Grundidee
#
# **Bei normalen Zahlen:**
# - $5 × \frac{1}{5} = 1$
# - $\frac{1}{5}$ ist die Inverse von 5
#
# **Bei Matrizen:**
# - $A × A^{-1} = I$
# - $A^{-1}$ ist die Inverse von $A$
# - Das Ergebnis ist die Identitätsmatrix!

# %% [markdown]
#
# ## Matrix-Inverse berechnen

# %%
A = np.array([[4, 7], [2, 6]])

# %%

# %% [markdown]
#
# Berechnen der Inversen

# %%

# %%

# %% [markdown]
#
# ### Verifikation: $A × A^{-1} = I$
#
# - Multipliziere A mit seiner Inversen

# %%

# %%

# %% [markdown]
#
# Vergleich mit Identitätsmatrix

# %%

# %% [markdown]
#
# ### Auch andersherum: $A^{-1} × A = I$

# %%

# %%

# %%

# %% [markdown]
#
# ## Größere Matrizen invertieren

# %%
B = np.array([[2, 1, 1], [1, 3, 2], [1, 0, 0]])

# %%

# %%

# %%

# %% [markdown]
#
# Verifikation:

# %%

# %%

# %%

# %% [markdown]
#
# ## Matrix-Inversion als "Division"
#
# **Problem:** Man kann Matrizen nicht direkt dividieren
#
# **Lösung:** Mit der Inversen multiplizieren
#
# Statt $A / B$ (nicht definiert) →  $A × B^{-1}$

# %% [markdown]
#
# ### Beispiel: Gleichung lösen
#
# Gegeben: $A × x = b$
#
# Gesucht: $x$
#
# Lösung:
# - Multipliziere beide Seiten mit $A^{-1}$
#
# $$
# \begin{align*}
# A^{-1} × A × x &= A^{-1} × b \\
# I × x &= A^{-1} × b \\
# x &= A^{-1} × b
# \end{align*}
# $$

# %% [markdown]
#
# Konkretes Beispiel:
#
# $A \times x = b$ von oben mit
#
# $$
# A = \begin{bmatrix} 3 & 2 \\ 4 & 1 \end{bmatrix}, \quad
# b = \begin{bmatrix} 7 \\ 6 \end{bmatrix}
# $$

# %%
A = np.array([[3, 2], [4, 1]])

# %%
b = np.array([7, 6])

# %%


# %% [markdown]
#
# Lösung mit Inverser:

# %%

# %%

# %%

# %% [markdown]
#
# Verifikation: Setze x in ursprüngliche Gleichung ein

# %%

# %%

# %%

# %% [markdown]
#
# ## Visualisierung: Was macht eine Inverse?
#
# - Eine Matrix $A$ transformiert Vektoren
# - Die Inverse $A^{-1}$ macht diese Transformation rückgängig

# %%
v = np.array([1, 0])

# %%
A = np.array([[2, 1], [1, 2]])

# %% [markdown]
#
# ### Visualisierung der Transformation und ihrer Inverse

# %%
plot_matrix_inverse_transformation(v, A)

# %%
plot_matrix_inverse_transformation(np.array([0.3, 1]), [[-3, 3.5], [0.6, 1.5]])

# %% [markdown]
#
# ## Wichtige Eigenschaften
#
# **1. Nicht alle Matrizen sind invertierbar:**
# - Matrix muss quadratisch sein ($n×n$)
# - Matrix muss "vollen Rang" haben (linear unabhängige Spalten)
# - Determinante ≠ 0

# %% [markdown]
#
# **2. Rechenregeln:**
# - $(A^{-1})^{-1} = A$
# - $(A × B)^{-1} = B^{-1} × A^{-1}$ (Reihenfolge umgekehrt!)
# - $(A^T)^{-1} = (A^{-1})^T$

# %% [markdown]
#
# ### Beispiele für nicht-invertierbare Matrizen
#
# - Matrix mit Null-Zeile (singulär)

# %%
singular = np.array([[1, 2], [0, 0]])

# %%
singular

# %%
try:
    np.linalg.inv(singular)
except np.linalg.LinAlgError as e:
    print(f"\nError: {e}")

# %% [markdown]
#
# - Matrix mit linear abhängigen Zeilen:
#   - Zeile 2 ist das Doppelte von Zeile 1

# %%
dependent = np.array([[1, 2], [2, 4]])

# %%
dependent

# %%
try:
    np.linalg.inv(dependent)
except np.linalg.LinAlgError as e:
    print(f"\nError: {e}")

# %% [markdown]
#
# **`lstsq` für robuste Lösungen:**
#
# - Löst lineare Systeme auch ohne exakte Lösung
# - Nutzt SVD statt Inversion
# - Wird in ML für Training verwendet!

# %%
A = np.array([[3, 2], [4, 1]])

# %%
b = np.array([7, 6])

# %%

# %%

# %%

# %%
A = np.array([[1, 2], [2, 4]])
b = np.array([5, 10])

# %%

# %%

# %%
A = np.array([[1, 2], [2, 4]])
b = np.array([5, 11])

# %%

# %%


# %% [markdown]
#
# ## Zusammenfassung
#
# **Identitätsmatrix:**
# - Quadratische Matrix mit Einsen auf der Diagonale
# - $A × I = I × A = A$
# - Verhält sich wie die Zahl 1
#
# **Matrix-Inverse:**
# - $A × A^{-1} = A^{-1} × A = I$
# - "Division" für Matrizen
# - Nicht alle Matrizen sind invertierbar
#

# %% [markdown]
#
# ## Workshop: Matrix-Inversion und Gleichungssysteme

# %% [markdown]
#
# ### Aufgabe 1: Identitätsmatrix verifizieren
#
# Erstellen Sie eine beliebige 3×3 Matrix und verifizieren Sie, dass die
# Multiplikation mit der Identitätsmatrix die Matrix nicht verändert.

# %% [markdown]
#
# ### Aufgabe 2: Inverse berechnen und verifizieren
#
# Gegeben ist die Matrix:
# ```python
# B = np.array([[1, 2],
#               [3, 4]])
# ```
#
# Berechnen Sie die Inverse und verifizieren Sie, dass $B × B^{-1} = I$.

# %% [markdown]
#
# ### Aufgabe 3: Gleichungssystem mit Inversion lösen
#
# Lösen Sie das folgende Gleichungssystem mit Matrix-Inversion:
#
# ```
# 2x + y = 5
# x + 3y = 7
# ```
#
# Hinweis: Schreiben Sie es als $A × x = b$ und lösen mit $x = A^{-1} × b$

# %% [markdown]
#
# ### Aufgabe 4: Gleichungssystem mit `lstsq` lösen
#
# Lösen Sie dasselbe System wie in Aufgabe 3, aber diesmal mit `np.linalg.lstsq()`.
# Vergleichen Sie das Ergebnis mit der Lösung durch Inversion.
#
# System:
# ```
# 2x + y = 5
# x + 3y = 7
# ```
