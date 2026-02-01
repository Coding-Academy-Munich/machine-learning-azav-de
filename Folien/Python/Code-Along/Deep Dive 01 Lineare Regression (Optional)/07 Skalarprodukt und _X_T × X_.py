# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Skalarprodukt und $X^T × X$</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Skalarprodukt und $X^T × X$
#
# Eine wichtige Beziehung: Das Skalarprodukt von Spaltenvektoren!
#
# Für zwei Spaltenvektoren $v$ und $w$:
# $$v^T × w = \text{Skalarprodukt}$$

# %%
import numpy as np
from display_utils import side_by_side, with_shape

# %% [markdown]
#
# ### Skalarprodukt mit Spaltenvektoren
#
# Das Skalarprodukt zweier Vektoren: $$v \cdot w = v_1 w_1 + v_2 w_2 + v_3 w_3 +
# \ldots$$
#
# - Wie sieht das als Multiplikation zweier Spaltenvektoren aus?
#   - $v$ und $w$ sind (3,1) Matrizen
#   - Wir müssen $v$ zu einer (1,3) Matrix transponieren, damit die Dimensionen
#     passen
#   - Dann ergibt die Matrix-Multiplikation das Skalarprodukt

# %%
v = np.array([[1], [2], [3]])
w = np.array([[4], [5], [6]])

# %%

# %%

# %% [markdown]
#
# Als Matrix-Multiplikation:
# $$v^T × w = \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix} × \begin{bmatrix} w_1 \\ w_2 \\ w_3 \end{bmatrix}$$
# $v^T$ ist ein Zeilenvektor.

# %%

# %%

# %% [markdown]
#
# - Das Matrixprodukt von $v^T$ und $w$ ergibt das Skalarprodukt
# - Allerdings als (1,1) Matrix, nicht als Vektor

# %%

# %%

# %%

# %% [markdown]
#
# Vergleich mit NumPy's `dot()`:
#
# - Konvertieren der Spaltenvektoren zu (NumPy) Vektoren (1D-Arrays)

# %%

# %%

# %% [markdown]
#
# Beide Berechnungen ergeben das gleiche Ergebnis.

# %%

# %% [markdown]
#
# ## $X^T × X$ für Matrizen
#
# Wenn $X$ eine Matrix mit Spaltenvektoren ist:
# $$X = \begin{bmatrix} | & | & | \\ v_1 & v_2 & v_3 \\ | & | & | \end{bmatrix}$$
#
# Dann ist $X^T × X$ eine Matrix aller Skalarprodukte:

# %% [markdown]
#
# $$X^T × X = \begin{bmatrix}
# v_1^T × v_1 & v_1^T × v_2 & v_1^T × v_3 \\
# v_2^T × v_1 & v_2^T × v_2 & v_2^T × v_3 \\
# v_3^T × v_1 & v_3^T × v_2 & v_3^T × v_3
# \end{bmatrix}$$
#
# - Diagonale: Skalarprodukte der Vektoren mit sich selbst
# - Außerhalb: Skalarprodukte zwischen verschiedenen Vektoren

# %% [markdown]
#
# ### Beispiel mit konkreten Zahlen

# %%
# Matrix with 3 column vectors
X = np.array([[1, 2, 1],
              [2, 1, 0],
              [3, 3, 2]])

# %%
# %with_shape X

# %% [markdown]
#
# Berechne $X^T × X$:

# %%

# %%


# %% [markdown]
#
# Überprüfen wir die Einträge manuell:

# %%
# Extract column vectors
v1 = X[:, 0:1]  # Keep as column vector
v2 = X[:, 1:2]
v3 = X[:, 2:3]

# %%
# %side_by_side v1, v2, v3


# %% [markdown]
#
# Diagonaleinträge (Vektoren mit sich selbst):

# %%
# v1 • v1
v1_dot_v1 = v1.T @ v1

# %%
# v2 • v2
v2_dot_v2 = v2.T @ v2

# %%
# v3 • v3
v3_dot_v3 = v3.T @ v3

# %%

# %% [markdown]
#
# Außerdiagonale Einträge (verschiedene Vektoren):

# %%
# v1 • v2
v1_dot_v2 = v1.T @ v2

# %%
# v1 • v3
v1_dot_v3 = v1.T @ v3

# %%
# v2 • v3
v2_dot_v3 = v2.T @ v3

# %%
# %side_by_side v1_dot_v2, v1_dot_v3, v2_dot_v3

# %% [markdown]
#
# Vergleichen mit $X^T × X$:

# %%
print("X^T × X:")
print(XTX)

# %%
print("\nManual calculation:")
print(f"Position (0,0): v1•v1 = {v1_dot_v1[0,0]}")
print(f"Position (0,1): v1•v2 = {v1_dot_v2[0,0]}")
print(f"Position (0,2): v1•v3 = {v1_dot_v3[0,0]}")
print(f"Position (1,0): v2•v1 = {v1_dot_v2[0,0]}")  # Symmetric
print(f"Position (1,1): v2•v2 = {v2_dot_v2[0,0]}")
print(f"Position (1,2): v2•v3 = {v2_dot_v3[0,0]}")
print(f"Position (2,0): v3•v1 = {v1_dot_v3[0,0]}")  # Symmetric
print(f"Position (2,1): v3•v2 = {v2_dot_v3[0,0]}")  # Symmetric
print(f"Position (2,2): v3•v3 = {v3_dot_v3[0,0]}")

# %% [markdown]
#
# ## Wichtige Eigenschaften von $X^T × X$
#
# 1. **Symmetrisch:** $(X^T X)^T = X^T X$
#    - Weil $v_i^T × v_j = v_j^T × v_i$ (Kommutativität des Skalarprodukts)
#
# 2. **Quadratisch:** Wenn $X$ Form $(n, m)$ hat, dann hat $X^T X$ Form $(m, m)$
#
# 3. **Positiv semidefinit:** Alle Eigenwerte sind $\geq 0$

# %% [markdown]
#
# ### Symmetrie überprüfen

# %%

# %%

# %%

# %% [markdown]
#
# ## Anwendung in Machine Learning
#
# $X^T × X$ erscheint häufig in ML-Formeln:
#
# - **Lineare Regression:** Normalengleichung
#   $$\theta = (X^T X)^{-1} X^T y$$
#
# - **Kovarianzmatrix:** (wenn $X$ zentriert ist)
#   $$\text{Cov}(X) = \frac{1}{n-1} X^T X$$
#
# - Misst, wie stark Features miteinander korrelieren

# %% [markdown]
#
# ### Intuition
#
# Die Diagonaleinträge von $X^T X$ zeigen:
# - Wie "groß" jeder Feature-Vektor ist
# - Die Summe der Quadrate der Feature-Werte
#
# Die Außerdiagonaleinträge zeigen:
# - Wie ähnlich verschiedene Features sind
# - Große Werte → Features sind korreliert

# %% [markdown]
#
# ## Workshop: $X^T × X$ und seine Eigenschaften

# %% [markdown]
#
# ### Aufgabe 1: Berechnung von $X^T × X$
#
# Gegeben ist die Matrix:
# ```python
# X = np.array([[1, 3],
#               [2, 1],
#               [1, 2],
#               [3, 1]])
# ```
#
# 1. Berechnen Sie $X^T × X$
# 2. Was ist die Form des Ergebnisses?
# 3. Interpretieren Sie die Diagonaleinträge

# %% [markdown]
#
# ### Aufgabe 2: Symmetrie überprüfen
#
# Gegeben ist die Matrix:
# ```python
# A = np.array([[1, 2, 0],
#               [3, 1, 4],
#               [2, 2, 1],
#               [0, 3, 2]])
# ```
#
# 1. Berechnen Sie $A^T × A$
# 2. Überprüfen Sie, ob das Ergebnis symmetrisch ist
# 3. Vergleichen Sie $(A^T A)^T$ mit $A^T A$

# %% [markdown]
#
# ### Aufgabe 3: Manuelle Überprüfung
#
# Gegeben ist die Matrix:
# ```python
# B = np.array([[2, 1],
#               [1, 3],
#               [0, 2]])
# ```
#
# 1. Berechnen Sie $B^T × B$
# 2. Extrahieren Sie die beiden Spaltenvektoren von B
# 3. Berechnen Sie manuell:
#    - $v_1^T × v_1$ (sollte Position (0,0) ergeben)
#    - $v_1^T × v_2$ (sollte Position (0,1) ergeben)
#    - $v_2^T × v_2$ (sollte Position (1,1) ergeben)
# 4. Vergleichen Sie mit $B^T × B$

# %% [markdown]
#
# ### Aufgabe 4: Form und Dimensionen
#
# Für verschiedene Matrizen, bestimmen Sie die Form von $X^T × X$:
#
# 1. $X$ hat Form $(5, 3)$ → Was ist die Form von $X^T × X$?
# 2. $X$ hat Form $(10, 4)$ → Was ist die Form von $X^T × X$?
# 3. $X$ hat Form $(n, m)$ → Was ist die Form von $X^T × X$?
#
# Überprüfen Sie Ihre Antworten mit konkreten Beispielen.

# %% [markdown]
#
# ### Aufgabe 5: Eigenwerte (Bonus)
#
# Gegeben ist die Matrix:
# ```python
# C = np.array([[1, 0],
#               [2, 1],
#               [1, 2]])
# ```
#
# 1. Berechnen Sie $C^T × C$
# 2. Berechnen Sie die Eigenwerte von $C^T × C$
# 3. Überprüfen Sie, ob alle Eigenwerte $\geq 0$ sind
#
# **Hinweis:** Verwenden Sie `np.linalg.eigvals()` für die Eigenwerte.
