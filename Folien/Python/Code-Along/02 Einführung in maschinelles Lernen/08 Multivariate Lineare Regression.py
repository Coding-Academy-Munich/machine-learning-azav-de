# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Multivariate Lineare Regression</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Lineare Regression mit mehreren Features
#
# - Bei fast allen Machine Learning Projekten hängen die Labels von mehreren
#   Features ab.
# - Eine lineare Regression mit mehreren Features nennt man multivariate lineare
#   Regression.

# %% [markdown]
#
# - Im folgenden Beispiel wollen wir die Note eines Studenten in der
#   Abschlussprüfung vorhersagen.
# - Dabei beziehen wir folgende Features mit ein:
#   - Note in der Zwischenprüfung
#   - Anzahl der Stunden, die für das Lernen aufgewendet wurden
#   - Anzahl der versäumten Kurstage

# %% [markdown]
#
# Die Daten sind in der sqlite Datenbank `student_grades.db` gespeichert.
#
# - Laden Sie die Daten aus der Datenbank in "DB Browser for SQLite" und
#   analysieren Sie die Struktur der Daten.
# - Verwenden Sie anschließend die `sqlite3` Bibliothek, um die Daten aus der
#   Datenbank zu laden.
#   - Erzeugen Sie eine Liste `student_x` mit den Features als Listen von
#     numerischen Werten
#   - Erzeugen Sie eine Liste `student_y` mit den Labels als numerischen Werten

# %% [markdown]
#
# Mit dem DB Browser sehen wir, dass die Daten in einer Tabelle `grades` mit den
# Spalten `midterm_grade`, `study_hours`, `missed_classes` und `final_grade`
# gespeichert sind.
#
# - Um die `student_x` Liste zu erzeugen, verwenden wir die Spalten
#   `midterm_grade`, `study_hours` und `missed_classes`.

# %%
import sqlite3

# %% [markdown]
#
# Erstellen einer Datenbankverbindung und eines Cursors

# %%

# %%

# %% [markdown]
#
# Ausführen der `SELECT` Anweisung

# %%

# %% [markdown]
#
# Erstellen einer Liste mit allen Ergebnis-Daten und Schließen der Verbindung

# %%

# %%

# %%

# %% [markdown]
#
# Erstellen der `student_x` und `student_y` Listen

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Plotten Sie die Daten in Scatterplots, um die Beziehungen zwischen den
# einzelnen Features und den Labels zu visualisieren.

# %%

# %% [markdown]
#
# Erstellen der Scatterplots mit Matplotlib

# %%

# %%

# %%

# %% [markdown]
#
# ### Histogramme
#
# - Ein Histogramm zeigt die Verteilung eines einzelnen Features
# - Nützlich, um Ausreißer und die allgemeine Verteilung der Daten zu
#   erkennen
# - Kann mit Matplotlib erstellt werden

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Seaborn ist eine Erweiterung von Matplotlib mit
#  - ansprechenden Standard-Stil-Optionen
#  - erweiterte Plot-Funktionen

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Seaborn bietet komplexere Diagrammtypen an, z.B. `regplot`

# %%

# %%

# %%

# %% [markdown]
#
# Seaborn `pairplot`:
#
# - Zeigt Beziehungen zwischen allen Features in einem Datensatz
# - Benötigt einen Pandas `DataFrame` als Eingabe
# - Hier nur als Beispiel, ohne genauere Erklärung

# %%

# %%

# %%

# %% [markdown]
#
# Trainieren Sie nun ein lineares Regressionsmodell mit den Daten in `student_x`
# und `student_y`.
#
# - Ein solches Modell mit mehreren Features wird auch als multivariates lineares
#   Modell bezeichnet.
# - Das Ergebnis wird immer mit der Formel $$y = m_1 x_1 + m_2 x_2 + ... + m_n x_n + b$$ berechnet
#   - $x_1, x_2, ..., x_n$ sind die Features,
#   - $m_1, m_2, ..., m_n$ die Koeffizienten (Steigungen) und
#   - $b$ der Achsenabschnitt.

# %% [markdown]
#
# Train-Test-Split der Daten

# %%

# %%

# %%

# %% [markdown]
#
# Trainieren des Modells

# %%

# %%

# %% [markdown]
#
# Bewerten Sie das Modell mit
# - Mean Absolute Error (MAE)
# - Mean Squared Error (MSE)
# - Bestimmtheitsmaß ($R^2$-Wert)

# %%

# %%

# %%

# %%

# %%


# %%

# %%

# %%

# %% [markdown]
#
# Plotten Sie die Vorhersagen gegen die tatsächlichen Werte in einem
# Scatterplot.

# %%

# %% [markdown]
#
# Erstellen Sie ein Histogramm der Vorhersagefehler für die Testdaten.

# %%

# %%

# %% [markdown]
#
# ## Workshop: Eiscreme
#
# In diesem Workshop werden wir den Verkauf von Eiscreme basierend auf der
# Temperatur und der Bewölkung vorhersagen.

# %% [markdown]
#
# Die Daten sind in der sqlite Datenbank `ice_cream_sales.db` gespeichert.
#
# - Laden Sie die Daten aus der Datenbank in "DB Browser for SQLite" und
#   analysieren Sie die Struktur der Daten.
# - Verwenden Sie anschließend die `sqlite3` Bibliothek, um die Daten aus der
#   Datenbank zu laden.
#   - Erzeugen Sie eine Liste `ice_cream_x` mit den Features als Listen von
#     numerischen Werten
#   - Erzeugen Sie eine Liste `ice_cream_y` mit den Labels als numerischen
#     Werten

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Plotten Sie die Daten in Scatterplots, um die Beziehungen zwischen den
# einzelnen Features und den Labels zu visualisieren.

# %%

# %%

# %%

# %%

# %%


# %% [markdown]
#
# Trainieren Sie nun ein lineares Regressionsmodell mit den Daten in
# `ice_cream_x` und `ice_cream_y`.

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Bewerten Sie das Modell mit
# - Mean Absolute Error (MAE)
# - Mean Squared Error (MSE)
# - Bestimmtheitsmaß ($R^2$-Wert)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Plotten Sie die Vorhersagen gegen die tatsächlichen Werte in einem
# Scatterplot.

# %%

# %%

# %% [markdown]
#
# Erstellen Sie ein Histogramm der Vorhersagefehler für die Testdaten.

# %%

# %%
