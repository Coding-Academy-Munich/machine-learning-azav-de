# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Training in der Praxis</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Training in der Praxis
#
# - Wir verstehen jetzt **wie** neuronale Netze lernen
# - Loss-Funktion + Gradientenabstieg
# - Aber: Es gibt viele praktische Aspekte zu beachten
# - Schauen wir uns diese genauer an

# %% [markdown]
#
# ## Epochen in der Praxis
#
# - Eine **Epoche** = einmal durch alle Trainingsdaten
# - **Zu wenig**: Netz hat nicht genug gelernt (Underfitting)
# - **Zu viel**: Netz lernt Trainingsdaten auswendig (Overfitting)
# - Wie finden wir die richtige Anzahl?

# %% [markdown]
#
# ## Beispieldaten: Make Moons

# %%
from nn_training_essentials_plots import plot_moons_data

# %%

# %% [markdown]
#
# ## Experiment: Verschiedene Epochen-Anzahlen
#
# - Training vs. Test Genauigkeit
# - Beobachten Sie den Unterschied!

# %%
from nn_training_essentials_plots import plot_epoch_experiment

# %%

# %% [markdown]
#
# ## Was sehen wir?
#
# - Training-Genauigkeit steigt (fast) immer
# - Test-Genauigkeit erreicht irgendwann ein Maximum
# - Danach: **Overfitting** - das Netz lernt Rauschen auswendig
# - **Lösung**: Stoppen, wenn Test-Genauigkeit nicht mehr steigt

# %% [markdown]
#
# ## Lernrate (Learning Rate)
#
# - Wie groß sind die Anpassungsschritte?
# - **Zu klein**: Lernen dauert ewig
# - **Zu groß**: Das Netz "springt" über die Lösung
# - **Goldlöckchen-Prinzip**: Nicht zu groß, nicht zu klein!

# %% [markdown]
#
# ## Experiment: Verschiedene Lernraten

# %%
from nn_training_essentials_plots import plot_learning_rate_curves

# %%

# %%

# %% [markdown]
#
# ## Batch Size
#
# - **Batch Size**: Wie viele Beispiele auf einmal verarbeiten?
# - **Vollständiger Batch**: Alle Daten auf einmal - langsam aber stabil
# - **Mini-Batch**: Kleine Gruppen - schneller, etwas rauschiger
# - **Einzelnes Beispiel (SGD)**: Sehr schnell, sehr rauschig
# - Üblich: Mini-Batches von 32, 64, 128 Beispielen

# %% [markdown]
#
# ## Early Stopping
#
# - Problem: Woher wissen wir, wann wir aufhören sollen?
# - **Early Stopping**: Stoppe, wenn Test-Fehler nicht mehr besser wird
# - Verhindert Overfitting
# - sklearn macht das automatisch (wenn `early_stopping=True`)

# %% [markdown]
#
# ## Regularisierung: Overfitting vermeiden
#
# - **Regularisierung**: Verhindert zu komplexe Modelle
# - Parameter: `alpha` (höher = stärkere Regularisierung)
# - **Zu klein (`alpha=0`)**: Overfitting möglich
# - **Zu groß**: Underfitting
# - Standard: `alpha=0.0001` (funktioniert oft gut)

# %% [markdown]
#
# ## Vergleich verschiedener `alpha`-Werte

# %%
from nn_training_essentials_plots import plot_alpha_comparison

# %%

# %% [markdown]
#
# ## Wann ist mein Modell überangepasst?
#
# - **Training-Fehler** ist sehr klein
# - **Test-Fehler** ist deutlich größer
# - Das Netz hat die Trainingsdaten auswendig gelernt
# - **Lösung**: Mehr Daten, Regularisierung, oder Early Stopping

# %% [markdown]
#
# ## Beispiel: Overfitting erkennen

# %%
from nn_training_essentials_plots import plot_overfitting_example

# %%

# %% [markdown]
#
# ## Praktische Tipps
#
# ### Anzahl Epochen
# - Start: 100-200 Epochen
# - sklearn stoppt automatisch bei fehlendem Fortschritt
#
# ### Lernrate
# - sklearn wählt meist gute Standardwerte
# - Bei Problemen: Dokumentation prüfen

# %% [markdown]
#
# ## Praktische Tipps (Fortsetzung)
#
# ### Regularisierung
# - Start mit `alpha=0.0001` (sklearn Standard)
# - Bei Overfitting: `alpha` erhöhen (z.B. 0.001, 0.01)
# - Bei Underfitting: `alpha` verringern (z.B. 0.00001)
#
# ### Wichtigste Regel
# - **Immer** Training- UND Test-Fehler beobachten!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Epochen**: Wie oft durch die Daten (zu viele → Overfitting)
# - **Lernrate**: Schrittgröße beim Lernen (Goldlöckchen-Prinzip!)
# - **Batch Size**: Wie viele Beispiele gleichzeitig
# - **Early Stopping**: Automatisch stoppen bei Overfitting
# - **Regularisierung (alpha)**: Verhindert zu komplexe Modelle
#
# **Wichtigste Regel**: Trainings- UND Test-Fehler beobachten!
