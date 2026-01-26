# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Coroutinen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Coroutinen
#
# Bisher haben Generatoren nur Werte **ausgegeben**.
# Aber was, wenn sie auch Werte **empfangen** könnten?

# %% [markdown]
#
# ## Zwei-Wege-Kommunikation
#
# - `next(gen)` → Generator gibt einen Wert zurück
# - `gen.send(value)` → Wir senden einen Wert **in** den Generator
#
# Ein Generator der `send()` verwendet wird **Coroutine** genannt.
# Coroutinen sind nützlich für:
# - Kooperatives Multitasking
# - Event-basierte Programmierung
# - Zustandsmaschinen

# %% [markdown]
#
# ## Wie funktioniert `send()`?
#
# - Der `yield`-Ausdruck kann einen **Wert empfangen**
# - `x = yield` speichert den gesendeten Wert in `x`
# - Der erste Aufruf muss `send(None)` sein (um den Generator zu starten)

# %% [markdown]
#
# ## Beispiel: Einfache Coroutine
#
# Diese Coroutine empfängt Werte und gibt sie aus:


# %%

# %%

# %% [markdown]
#
# **Schritt 1:** Generator starten mit `send(None)`:

# %%

# %% [markdown]
#
# **Schritt 2:** Werte senden:

# %%

# %%

# %% [markdown]
#
# **Schritt 3:** Nach `n` Werten ist die Coroutine erschöpft:

# %%


# %% [markdown]
#
# ## Coroutine die Werte sendet UND empfängt
#
# `yield` kann gleichzeitig einen Wert zurückgeben und einen empfangen:
#
# ```python
# x = yield value  # gibt 'value' zurück, empfängt in 'x'
# ```

# %%

# %%

# %% [markdown]
#
# `send(None)` startet und gibt den ersten Wert zurück:

# %%

# %% [markdown]
#
# Jetzt können wir Werte hin- und herschicken:

# %%


# %% [markdown]
#
# ## Praktisches Beispiel: Zähler mit Reset
#
# Ein Zähler, der hochzählt, aber per `send()` zurückgesetzt werden kann:

# %%

# %%

# %% [markdown]
#
# Normal hochzählen mit `next()`:

# %%

# %%

# %% [markdown]
#
# Per `send()` auf einen neuen Wert setzen:

# %%

# %%

# %% [markdown]
#
# Nach dem Maximum ist der Generator erschöpft:

# %%
