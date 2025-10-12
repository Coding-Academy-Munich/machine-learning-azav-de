# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: Eiscreme</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Vorbereitung: Mehrere Features
#
# Bei fast allen Machine Learning Projekten hängen die Labels von mehreren
# Features ab.
#
# Im folgenden Beispiel wollen wir die Note eines Studenten in der
# Abschlussprüfung vorhersagen. Dabei beziehen wir folgende Labels mit ein:
#
# - Note in der Zwischenprüfung
# - Anzahl der Stunden, die für das Lernen aufgewendet wurden
# - Anzahl der versäumten Kurstage

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

# %%
connection = sqlite3.connect("student_grades.db")
cursor = connection.cursor()

# %%
cursor.execute(
    "SELECT midterm_grade, study_hours, missed_classes, final_grade FROM grades"
)
rows = cursor.fetchall()
connection.close()

# %%
rows[:5]

# %%
student_x = [list(row[:3]) for row in rows]

# %%
student_x[:5]

# %%
student_y = [row[3] for row in rows]

# %%
student_y[:5]

# %% [markdown]
#
# Plotten Sie die Daten in Scatterplots, um die Beziehungen zwischen den
# einzelnen Features und den Labels zu visualisieren.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
plt.scatter([x[0] for x in student_x], student_y)
plt.xlabel("Midterm Grade")
plt.ylabel("Final Grade")
plt.show()

# %%
plt.scatter([x[1] for x in student_x], student_y)
plt.xlabel("Study Hours")
plt.ylabel("Final Grade")
plt.show()

# %%
plt.scatter([x[2] for x in student_x], student_y)
plt.xlabel("Missed Classes")
plt.ylabel("Final Grade")
plt.show()

# %%
sns.regplot(x=[x[0] for x in student_x], y=student_y)
plt.xlabel("Midterm Grade")
plt.ylabel("Final Grade")
plt.show()

# %%
sns.regplot(x=[x[1] for x in student_x], y=student_y)
plt.xlabel("Study Hours")
plt.ylabel("Final Grade")
plt.show()

# %%
sns.regplot(x=[x[2] for x in student_x], y=student_y)
plt.xlabel("Missed Classes")
plt.ylabel("Final Grade")
plt.show()

# %% [markdown]
#
# Trainieren Sie nun ein lineares Regressionsmodell mit den Daten in `student_x`
# und `student_y`.

# %%
from sklearn.model_selection import train_test_split

# %%
student_x_train, student_x_test, student_y_train, student_y_test = train_test_split(
    student_x, student_y, test_size=0.4, random_state=42
)

# %%
student_x_train[:5], student_y_train[:5]

# %%
from sklearn.linear_model import LinearRegression

# %%
model = LinearRegression()
model.fit(student_x_train, student_y_train)

# %% [markdown]
#
# Bewerten Sie das Modell mit dem Mean Absolute Error (MAE) und dem Mean Squared
# Error (MSE).

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
train_predictions = model.predict(student_x_train)
test_predictions = model.predict(student_x_test)

# %%
mean_absolute_error(student_y_train, train_predictions)

# %%
mean_squared_error(student_y_train, train_predictions)


# %%
mean_absolute_error(student_y_test, test_predictions)

# %%
mean_squared_error(student_y_test, test_predictions)

# %% [markdown]
#
# Plotten Sie die Vorhersagen gegen die tatsächlichen Werte in einem
# Scatterplot.

# %%
plt.scatter(student_y_train, train_predictions, alpha=0.5)
plt.scatter(student_y_test, test_predictions, alpha=0.5)
plt.plot([0, 100], [0, 100], color="red", linestyle="--")
plt.xlabel("True Final Grade")
plt.ylabel("Predicted Final Grade")
plt.show()

# %%
sns.regplot(x=student_y_test, y=test_predictions)
plt.xlabel("True Final Grade")
plt.ylabel("Predicted Final Grade")
plt.show()

# %%
error = [true - pred for true, pred in zip(student_y_test, test_predictions)]
plt.hist(error, bins=10)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
#
# ## Workshop: Eiscreme
#
# In diesem Workshop werden wir ein kleines Machine Learning Projekt
# durchführen.
#
# Wir werden den Verkauf von Eiscreme basierend auf der Temperatur und der
# Bewölkung vorhersagen.

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
import sqlite3

# %%
connection = sqlite3.connect("ice_cream_sales.db")
cursor = connection.cursor()

# %%
cursor.execute("SELECT temperature, cloud_cover, sales FROM sales")
data = cursor.fetchall()
connection.close()

# %%
ice_cream_x = [list(row[:2]) for row in data]
ice_cream_y = [row[2] for row in data]

# %%
ice_cream_x[:5]

# %%
ice_cream_y[:5]

# %% [markdown]
#
# Plotten Sie die Daten in Scatterplots, um die Beziehungen zwischen den
# einzelnen Features und den Labels zu visualisieren.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
plt.scatter([x[0] for x in ice_cream_x], ice_cream_y)
plt.xlabel("Temperature")
plt.ylabel("Ice Cream Sales")
plt.show()

# %%
plt.scatter([x[1] for x in ice_cream_x], ice_cream_y)
plt.xlabel("Cloud Cover")
plt.ylabel("Ice Cream Sales")
plt.show()

# %%
sns.regplot(x=[x[0] for x in ice_cream_x], y=ice_cream_y)
plt.xlabel("Temperature")
plt.ylabel("Ice Cream Sales")
plt.show()

# %%
sns.regplot(x=[x[1] for x in ice_cream_x], y=ice_cream_y)
plt.xlabel("Cloud Cover")
plt.ylabel("Ice Cream Sales")
plt.show()


# %% [markdown]
#
# Trainieren Sie nun ein lineares Regressionsmodell mit den Daten in
# `ice_cream_x` und `ice_cream_y`.

# %%
from sklearn.model_selection import train_test_split

# %%
ice_cream_x_train, ice_cream_x_test, ice_cream_y_train, ice_cream_y_test = train_test_split(
    ice_cream_x, ice_cream_y, test_size=0.4, random_state=42
)

# %%
from sklearn.linear_model import LinearRegression

# %%
model = LinearRegression()
model.fit(ice_cream_x_train, ice_cream_y_train)

# %% [markdown]
#
# Bewerten Sie das Modell mit dem Mean Absolute Error (MAE) und dem Mean Squared
# Error (MSE).

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
train_predictions = model.predict(ice_cream_x_train)
test_predictions = model.predict(ice_cream_x_test)

# %%
mean_absolute_error(ice_cream_y_train, train_predictions)

# %%
mean_squared_error(ice_cream_y_train, train_predictions)

# %%
mean_absolute_error(ice_cream_y_test, test_predictions)

# %%
mean_squared_error(ice_cream_y_test, test_predictions)

# %% [markdown]
#
# Plotten Sie die Vorhersagen gegen die tatsächlichen Werte in einem
# Scatterplot.

# %%
plt.scatter(ice_cream_y_train, train_predictions, alpha=0.5)
plt.scatter(ice_cream_y_test, test_predictions, alpha=0.5)
plt.plot([0, 100], [0, 100], color="red", linestyle="--")
plt.xlabel("True Ice Cream Sales")
plt.ylabel("Predicted Ice Cream Sales")
plt.show()

# %%
sns.regplot(x=ice_cream_y_test, y=test_predictions)
plt.xlabel("True Ice Cream Sales")
plt.ylabel("Predicted Ice Cream Sales")
plt.show()

# %%
error = [true - pred for true, pred in zip(ice_cream_y_test, test_predictions)]
plt.hist(error, bins=20)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()

# %%
