# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Programm vs. Modell</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

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
# ## Ausführen eines Modells

# %%
from transformers import pipeline

# %%
sentiment_analysis_model = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

# %%
sentiment_analysis_model("I love machine learning!")

# %%
sentiment_analysis_model("I hate bad weather!")

# %%
sentiment_analysis_model("It's all right, I guess.")


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
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-06.png"
#      style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ## Programm vs. Modell
#
# <img src="img/program-vs-model-03.png"
#      style="width: 30%; margin-left: auto; margin-right: auto;"/>
# <img src="img/program-vs-model-06.png"
#      style="width: 40%; margin-left: auto; margin-right: auto;"/>
