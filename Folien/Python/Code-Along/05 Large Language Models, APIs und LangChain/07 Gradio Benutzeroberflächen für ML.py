# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Gradio: Benutzeroberflächen für ML</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Warum Benutzeroberflächen?
#
# - Kommandozeilen-Apps sind funktional, aber...
# - **Nicht benutzerfreundlich** für Nicht-Programmierer
# - **Nicht beeindruckend** im Portfolio
# - **Nicht teilbar** ohne Code
#
# **Lösung**: Einfache Web-Interfaces!

# %% [markdown]
#
# ## Was ist Gradio?
#
# - **Python-Bibliothek** für ML-Interfaces
# - **Wenige Zeilen Code** → Professionelle Web-UI
# - **Funktioniert mit jedem Python-Code**
# - **Teilbar** via Link
# - **Kostenlos und Open Source**
#
# **Ideal für**: Demos, Prototypen, Portfolio-Projekte

# %% [markdown]
#
# ## Installation
#
# ```bash
# pip install gradio
# ```
#
# **Das war's!** Keine weitere Konfiguration nötig.

# %%
# !pip install gradio --root-user-action=ignore

# %%
import gradio as gr

# %% [markdown]
#
# ## Erstes Beispiel: Hello World

# %%

# %%

# %%

# %% [markdown]
#
# ## Was passiert hier?
#
# 1. `fn=greet`: Unsere Python-Funktion
# 2. `inputs="text"`: Textfeld für Eingabe
# 3. `outputs="text"`: Textfeld für Ausgabe
# 4. `demo.launch()`: Startet Web-Server
#
# Gradio erstellt automatisch die Web-Oberfläche!
