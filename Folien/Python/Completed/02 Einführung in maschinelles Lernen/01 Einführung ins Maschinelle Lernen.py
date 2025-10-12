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
# ## Was ist Maschinelles Lernen (ML)?
#
# - Nicht so einfach abstrakt zu definieren
# - Betrachten wir ein Beispiel


# %% [markdown]
#
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-01.png"
#      style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-02.png"
#      style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ### Installation der benötigten Bibliotheken

# %%
# #!pip install matplotlib seaborn scikit-learn pandas numpy transformers huggingface_hub[hf_xet]

# %% [markdown]
#
# Installation von PyTorch mit CUDA-Unterstützung (falls GPU vorhanden)

# %%
# #!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# %% [markdown]
#
# Installation von PyTorch ohne CUDA-Unterstützung (CPU only)

# %%
# #!pip install torch torchvision

# %% [markdown]
#
# ## Ausführen eines Programms

# %%
def add(a, b):
    return a + b

# %%
add(2, 3)

# %% [markdown]
#
# ## Ausführen eines Modells

# %%
from transformers import pipeline

# %%
sentiment_analysis_model = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# %%
sentiment_analysis_model("I love machine learning!")

# %%
sentiment_analysis_model("I hate bad weather!")

# %%
sentiment_analysis_model("It's all right, I guess.")
