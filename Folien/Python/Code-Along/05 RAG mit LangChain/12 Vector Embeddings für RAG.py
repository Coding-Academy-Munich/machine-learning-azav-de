# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Vector Embeddings für RAG</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Rückblick: Embeddings
#
# - In Sektion 03 (Neuronale Netze) haben wir Embeddings kennengelernt
# - **Embeddings**: Wörter als Vektoren (Listen von Zahlen)
# - Ähnliche Wörter → ähnliche Vektoren
#
# **Für RAG**: Wir brauchen Embeddings für ganze Texte!

# %%
import os
import dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from langchain_openai import OpenAIEmbeddings
from vector_embeddings_plots import plot_2d_vectors

# %%
dotenv.load_dotenv()

# %% [markdown]
#
# ## Text-Embeddings
#
# - Nicht nur Wörter, sondern **längere Texte** einbetten
# - Sätze, Absätze, Dokumente → Vektoren
# - **Dimension**: Meist 384, 768, oder 1536
#   - Mehr Dimensionen = mehr Nuancen, aber auch teurer
# - **Modelle**:
#   - OpenAI: text-embedding-3-small (1536 Dimensionen)
#   - HuggingFace: verschiedene Open-Source-Modelle
# - **Vektor-Datenbanken** (z.B. Qdrant): speichern und durchsuchen Embeddings

# %% [markdown]
#
# ## Wahl des Embedding-Modells
#
# - Wir verwenden **`text-embedding-3-small`** von OpenAI
# - Gute Balance aus Qualität, Geschwindigkeit und Kosten
# - 1536 Dimensionen — ausreichend für die meisten Anwendungen
# - Unterstützt mehrsprachige Texte (Deutsch, Englisch, etc.)
# - Alternativen: `text-embedding-3-large` (genauer, teurer),
#   Open-Source-Modelle von HuggingFace (kostenlos, lokal ausführbar)
# - **Achtung**: Maximale Eingabelänge ist 8191 Tokens — längere Texte werden
#   stillschweigend abgeschnitten (daher ist gutes Chunking wichtig!)

# %% [markdown]
#
# ## Embeddings mit LangChain erstellen

# %% [markdown]
#
# Erstellen eines `OpenAIEmbeddings`-Modell mit OpenRouter:

# %%
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

# %%
texts = [
    "Machine Learning ist ein Teilbereich der KI",
    "Machine Learning ist eine wichtige KI-Methode",
    "Deep Learning nutzt neuronale Netze",
    "Katzen sind Haustiere"
]

# %% [markdown]
#
# Erstellen von Embeddings für alle Texte:

# %%
text_embeddings = embeddings.embed_documents(texts)

# %% [markdown]
# Anzahl Texte:

# %%

# %% [markdown]
# Dimensionen pro Embedding:

# %%

# %% [markdown]
# Erster Text:

# %%

# %% [markdown]
# Erste 5 Werte des Embeddings:

# %%

# %% [markdown]
#
# ## Semantische Ähnlichkeit
#
# - Embeddings ermöglichen **semantische Suche**
# - Nicht nur gleiche Wörter, sondern **ähnliche Bedeutung**
# - Aber wie messen wir die Ähnlichkeit zwischen zwei Vektoren?
# - Zwei Ansätze:
#   - **Euklidische Distanz**: Abstand zwischen zwei Punkten
#   - **Kosinus-Ähnlichkeit**: Winkel zwischen zwei Vektoren

# %% [markdown]
#
# ## Euklidische Distanz vs. Kosinus-Ähnlichkeit
#
# <div style="float:right;width:40%;">
#   <img src="img/cosine-distance.png" style="float:right;width:50%"/>
#   <img src="img/vector-difference.png" style="float:left;width:50%"/>
# </div>
# <div style="width:60%;">
# <br>
# <ul>
#   <li><b>Euklidische Distanz</b> (rechts): Abstand zwischen den Spitzen</li>
#   <ul>
#     <li>Misst |\(\vec a - \vec b\)| (Spitze zu Spitze)</li>
#     <li>Hängt von <b>Richtung und Länge</b> der Vektoren ab</li>
#   </ul>
#   <li><b>Kosinus-Ähnlichkeit</b> (links): Winkel zwischen den Pfeilen</li>
#   <ul>
#     <li>Misst nur die <b>Richtung</b>, ignoriert die Länge</li>
#     <li>\(\vec a\) und \(\vec b\): kleiner Winkel → hohe Ähnlichkeit</li>
#   </ul>
# </ul>
# </div>

# %% [markdown]
#
# ## Beispiel: 2D-Vektoren
#
# - Betrachten wir drei einfache 2D-Vektoren
# - $\vec a$ und $\vec b$: gleiche Richtung, aber $\vec b$ ist kürzer
# - $\vec c$: zeigt in eine ganz andere Richtung

# %%
a = np.array([[4, 3]])
b = np.array([[2, 1.5]])
c = np.array([[1, -3]])

# %%
plot_2d_vectors(a, b, c)

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Ergebnis: 2D-Beispiel
#
# - $\vec a$ und $\vec b$: **gleiche Richtung** (Kosinus-Ähnlichkeit = 1.0)
#   - Aber: Euklidische Distanz = 2.5 (wegen unterschiedlicher Länge)
# - $\vec a$ und $\vec c$: **verschiedene Richtungen** (Kosinus-Ähnlichkeit ≈ −0.45)
#   - Euklidische Distanz = 6.7 (weit entfernt)
# - Kosinus-Ähnlichkeit ignoriert die Länge und misst nur die Richtung
