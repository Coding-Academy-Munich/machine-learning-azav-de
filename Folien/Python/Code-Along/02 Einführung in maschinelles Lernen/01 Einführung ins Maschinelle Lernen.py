# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung ins Maschinelle Lernen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# <img src="img/robots-playing-checkers.png"
#      style="width: 90%; margin-left: auto; margin-right: auto; display: block;">

# %% [markdown]
#
# ### Was ist Maschinelles Lernen (ML)?
#
# Entwicklung von Systemen, die aus Daten lernen können

# %% [markdown]
#
# ### Warum?
#
# - Viele reale Probleme sind zu komplex, um sie mit traditionellen
#   Programmiertechniken zu lösen.
# - Die Funktionsweise eines Programms kann verbessert/angepasst werden,
#   indem wir ihm mehr Daten geben.

# %% [markdown]
#
# ### Wie funktioniert das?
#
# - Wir erstellen ein **Modell**
#   - Programm, das mehrere ähnliche Probleme lösen kann
# - Wir **trainieren** das Modell
#   - Indem wir ihm Daten zeigen, lernt es unser konkretes Problem zu lösen
# - Wir **evaluieren** das Modell
#   - Wir testen, wie gut es unser konkretes Problem löst

# %% [markdown]
#
# ### Beispiel: Preis eines Pizzas
#
# - Preis einer Pizza bestimmt durch
#   - Grundpreis
#   - Anzahl der Beläge (jeder Belag kostet gleich viel)
# - Ist das ein Problem, für das sich ML anbietet?

# %% [markdown]
#
# ### Programm zur Berechnung des Pizza-Preises

# %%

# %%

# %% [markdown]
#
# - Was ist, wenn wir das Programm für verschiedene Pizzerien verwenden wollen?
#   - Unterschiedliche Grundpreise
#   - Unterschiedliche Preise pro Belag
# - Wir können ein Programm schreiben, das für beliebige Preise funktioniert

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# - Unser `PizzaCalculator` kann "verschiedene Probleme" lösen
# - Aber er kann nicht "lernen"

# %% [markdown]
#
# ### Beispiel: Catering-Service
#
# - Wir wollen einen Catering-Service starten
# - Wie viel sollen wir für ein Event verlangen?
# - Vermutung: Konkurrenzpreise bestehen aus
#   - Grundpreis
#   - Preis pro Person
# - Aber wir bekommen von der Konkurrenz nur Angebote für komplette Events,
#   nicht die Komponenten dahinter
# - Wie können wir vorgehen?
#   - Ist das ein Problem, für das sich ML anbietet?

# %% [markdown]
#
# ### Vorgehensweise
#
# - Wir holen uns von Konkurrenten Angebote für Events mit verschiedenen
#   Personenzahlen
# - Daraus versuchen wir, den Grundpreis und den Preis pro Person zu
#   schätzen
#   - Werte ausprobieren
#   - Mathematisch: Lineare Regression
#   - Graphisch: Gerade durch Punktewolke legen

# %% [markdown]
#
# ### Daten von Anbieter A
#
# | Anzahl Personen | Angebotspreis |
# |-----------------|---------------|
# | 10              | 400 €         |
# | 20              | 700 €         |
# | 30              | 1000 €        |

# %% [markdown]
#
# ### Visualisierung der Preise

# %%
import matplotlib.pyplot as plt

# %%

# %%

# %%

# %% [markdown]
#
# ### Interaktive Visualisierung des Catering-Preismodells

# %%
from ipywidgets import interact, FloatSlider


# %%
def plot_catering_model(base_price=100.0, price_per_person=30.0):
    x_values = list(range(0, 41))

    model = CateringPriceModel(base_price, price_per_person)
    y_values = [model.predict_price(x) for x in x_values]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, "r-", linewidth=2, label="Model Prediction")
    plt.scatter(num_people, offered_prices, s=100, zorder=5, label="Provider A Data")
    plt.title(
        f"Catering Price Model (Base: €{base_price:.2f}, Per Person: €{price_per_person:.2f})"
    )
    plt.xlabel("Number of People")
    plt.ylabel("Price (€)")
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.xlim(0, 40)
    plt.ylim(0, max(max(y_values), max(offered_prices)) * 1.1)
    plt.show()


# %%
interact(
    plot_catering_model,
    base_price=FloatSlider(
        min=0, max=500, step=10, value=400, description="Base Price (€)"
    ),
    price_per_person=FloatSlider(
        min=0, max=100, step=5, value=10, description="Price/Person (€)"
    ),
)

# %% [markdown]
#
# Können wir das Berechnen von `base_price` und `price_per_person`
# automatisieren?

# %%

# %% [markdown]
#
# ### Trainieren des Modells

# %%

# %%

# %%

# %% [markdown]
#
# ### Verwenden des trainierten Modells
#
# Für Werte, die das Modell bereits gesehen hat:

# %%

# %% [markdown]
#
# Für Werte, die das Modell noch nie gesehen hat:

# %%

# %% [markdown]
#
# ### Verwenden von Scikit-Learn Modellen
#
# - Unser Modell hat nicht wirklich etwas mit "Catering" zu tun
#   - Es lässt sich auf jedes Problem anwenden, bei dem eine lineare Beziehung
#     zwischen Eingabe und Ausgabe besteht
# - Die Implementierung ist sehr einfach und nicht sehr robust
#   - Es ignoriert beim Training alle Punkte außer den ersten und letzten
#   - Es kann nicht mit Daten umgehen, die "Rauschen" enthalten
# - Die Bibliothek [Scikit-Learn](https://scikit-learn.org/) bietet
#   viele vorgefertigte Modelle, die wir verwenden können
# - Unser Modell nennt man eine **lineare Regression**

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ### Verwenden des Scikit-Learn Modells
#
# Für Werte, die das Modell bereits gesehen hat:

# %%

# %% [markdown]
#
# Für Werte, die das Modell noch nie gesehen hat:

# %%

# %% [markdown]
#
# Für mehrere Werte

# %%

# %% [markdown]
#
# ## Programm vs. trainiertes Modell
#
# <img src="img/program-vs-model-01.png"
#      style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. trainiertes Modell
#
# <img src="img/program-vs-model-02.png"
#      style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. Modell (Erstellung)
#
# <img src="img/program-vs-model-03.png"
#      style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. Modell (Erstellung)
#
# <img src="img/program-vs-model-04.png"
#      style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Workshop: Fremdenführer
#
# Sie wollen einen Service starten, der Touristen durch Ihre Stadt führt. Um die
# Preise für Ihre Führungen festzulegen, haben Sie die Preise von Wettbewerbern
# in der Nähe recherchiert und die folgenden Daten gesammelt:
#
# | Anzahl Personen | Angebotspreis |
# |-----------------|---------------|
# | 5               | 150 €         |
# | 10              | 250 €         |
# | 30              | 650 €         |
#
# - Erstellen Sie ein Modell, das den Angebotspreis Ihres Wettbewerbers in
#   Abhängigkeit von der Anzahl der Personen vorhersagt.
# - Sagt dieses Modell die bekannten Datenpunkte korrekt voraus?
# - Was sind die geschätzten Werte für Grundpreis und Preis pro Person?

# %%

# %% [markdown]
#
# Angenommen, Ihr Wettbewerber hätte Ihnen stattdessen die folgenden Preise
# genannt:
#
# | Anzahl Personen | Angebotspreis |
# |-----------------|---------------|
# | 5               | 150 €         |
# | 10              | 250 €         |
# | 30              | 500 €         |
#
# Könnten Sie die Preise mit dem gleichen Modell vorhersagen, oder gibt es dabei
# Probleme?
#
# Versuchen Sie, die Preise mit dem gleichen Modell vorherzusagen. Was fällt
# Ihnen auf? Können Sie das Problem erklären?


# %%

