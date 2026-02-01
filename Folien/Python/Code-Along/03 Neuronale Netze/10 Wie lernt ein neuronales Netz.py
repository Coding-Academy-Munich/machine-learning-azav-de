# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Wie lernt ein neuronales Netz?</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Rückblick: Unser erstes Netzwerk
#
# - Im letzten Thema haben wir ein neuronales Netz trainiert
# - Wir haben `mlp_model.fit(X_train, y_train)` aufgerufen
# - Das Netz hat gelernt, Vorhersagen zu machen
# - Aber **wie** lernt es eigentlich?

# %% [markdown]
#
# ## Das Lernziel
#
# - Beim Training hat das Netz ein klares Ziel
# - **Minimiere den Fehler** bei den Vorhersagen
# - Der Fehler misst, wie weit die Vorhersagen von den echten Werten entfernt sind
# - Je kleiner der Fehler, desto besser das Modell

# %% [markdown]
#
# ## Die Loss-Funktion
#
# - Der Fehler wird auch **Loss** genannt
# - Die **Loss-Funktion** berechnet, wie schlecht das Modell ist
# - Für Regression: oft MSE (Mean Squared Error)
# - Für Klassifikation: Cross-Entropy
# - **Ziel des Trainings**: Loss minimieren!

# %% [markdown]
#
# ## Was ist MSE?
#
# - **MSE** = Mean Squared Error (Mittlerer quadratischer Fehler)
# - Berechnung: $(y_{true} - y_{pred})^2$ für jeden Datenpunkt
# - Dann: Durchschnitt aller quadrierten Fehler
# - Große Fehler werden stärker bestraft (wegen Quadrat)

# %% [markdown]
#
# ## Epochen: Wie oft lernen wir?
#
# - Eine **Epoche** = einmal durch alle Trainingsdaten
# - Mehr Epochen = mehr Lerngelegenheiten
# - Aber: Irgendwann wird es nicht mehr besser
# - **Zu wenig**: Netz hat nicht genug gelernt
# - **Zu viel**: Netz lernt Trainingsdaten auswendig

# %% [markdown]
#
# ## Visualisierung: Fehler während des Trainings
#
# - Schauen wir uns an, wie sich der Fehler ändert
# - Beispiel: Vorhersage von Noten basierend auf Lernstunden

# %%
from nn_training_essentials_plots import plot_error_during_training

# %%

# %% [markdown]
#
# ## Was sehen wir?
#
# - Am Anfang: hoher Fehler (das Netz weiß noch nichts)
# - Während des Trainings: Fehler sinkt
# - Nach genug Training: Fehler stabilisiert sich
# - Das Netz hat gelernt!

# %% [markdown]
#
# ## Visualisierung: Vorhersagen verbessern sich
#
# - Schauen wir uns die Vorhersagen nach verschiedenen Epochen an
# - Von "keine Ahnung" zu "gute Vorhersage"

# %%
from nn_training_essentials_plots import plot_predictions_improving

# %%

# %% [markdown]
#
# ## Was passiert beim Lernen?
#
# - Das Netz hat **Parameter** (Gewichte und Bias-Werte)
# - Diese Parameter bestimmen die Vorhersagen
# - Beim Training werden diese Parameter angepasst
# - Ziel: Bessere Vorhersagen → kleinerer Fehler

# %% [markdown]
#
# ## Zusammenfassung
#
# - Neuronale Netze lernen durch **Minimierung des Fehlers**
# - Der Fehler (Loss) wird mit einer **Loss-Funktion** berechnet
# - Mehr Training (Epochen) → kleinerer Fehler
# - Die **Parameter** (Gewichte und Bias) werden angepasst
# - Aber **wie** werden sie angepasst?
# - Das sehen wir in der nächsten Lektion!

# %%
