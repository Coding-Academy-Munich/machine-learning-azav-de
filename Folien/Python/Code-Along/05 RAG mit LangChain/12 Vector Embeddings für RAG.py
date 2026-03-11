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
#
# Typ der Embeddings:

# %%

# %% [markdown]
# Anzahl Texte und Embeddings:

# %%

# %% [markdown]
# Dimensionen pro Embedding:

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
# - **Achtung**: Skalen sind invers — Distanz: 0 = identisch; Ähnlichkeit: 1.0
#   = identisch

# %% [markdown]
#
# ## Euklidische Distanz vs. Kosinus-Ähnlichkeit
#
# <img src="img/cosine-distance.png" style="float:right;width:20%"/>
# <img src="img/vector-difference.png" style="float:right;width:20%"/>
#
# - **Euklidische Distanz** (links): Abstand zwischen den Spitzen
#   - Misst $\|\vec a - \vec b\|$ (Spitze zu Spitze)
#   - Hängt von **Richtung und Länge** der Vektoren ab
# - **Kosinus-Ähnlichkeit** (rechts): Winkel zwischen den Pfeilen
#   - Misst nur die **Richtung**, ignoriert die Länge
#   - $\vec a$ und $\vec b$: kleiner Winkel → hohe Ähnlichkeit

# %% [markdown]
#
# ### Distanz vs. Ähnlichkeit
#
# Distanz und Ähnlichkeit sind verwandt, aber nicht dasselbe:
#
# - Euklidische Distanz:
#   - 0 = identisch
#   - höhere Werte = weniger ähnlich
# - Kosinus-Ähnlichkeit:
#   - 1.0 = identisch
#   - Werte nahe 0 = unverwandt
#   - Negative Werte = entgegengesetzt (bei Text-Embeddings selten)

# %% [markdown]
#
# ## Beispiel: 2D-Vektoren
#
# - Betrachten wir vier einfache 2D-Vektoren
# - $\vec a$ und $\vec b$: gleiche Richtung, aber $\vec b$ ist kürzer
# - $\vec c$: zeigt in eine ganz andere Richtung
# - $\vec d$: zeigt in eine ähnliche Richtung wie $\vec c$

# %%
a = np.array([[4, 3]])
b = np.array([[2, 1.5]])
c = np.array([[1.5, -2]])
d = np.array([[1, -3]])

# %%
plot_2d_vectors(a, b, c, d)

# %%

# %%

# %%

# %%

# %%
plot_2d_vectors(a, b, c, d)

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
# - $\vec a$ und $\vec c$: **90-Grad-Winkel** (Kosinus-Ähnlichkeit = 0.0)
#   - Euklidische Distanz = 5.6 (mittelgroßer Abstand)
# - $\vec a$ und $\vec d$: **verschiedene Richtungen** (Kosinus-Ähnlichkeit ≈
#   −0.3)
#   - Euklidische Distanz = 6.7 (weit entfernt)
# - Kosinus-Ähnlichkeit ignoriert die Länge und misst nur die Richtung
# - Denken Sie daran:
#   - **kleine** Distanz = ähnlich
#   - **hohe** Ähnlichkeit = ähnlich

# %% [markdown]
#
# ## Moderne Embedding-Modelle normalisieren!
#
# - Modelle wie `text-embedding-3-small` geben **normalisierte Vektoren** zurück
# - Alle Vektoren haben die **gleiche Länge** (Norm = 1), egal wie lang der Text
#   ist oder worüber er spricht
# - Für normalisierte Vektoren ergeben Euklidische Distanz und
#   Kosinus-Ähnlichkeit immer **dieselbe Rangfolge** der ähnlichsten Texte
#
# *Für Neugierige: Es gilt $d_{\text{euklid}} = \sqrt{2 \cdot (1 -
# \textit{cos\_sim})}$*

# %% [markdown]
#
# ## Warum trotzdem Kosinus-Ähnlichkeit?
#
# - **Konvention**: Die meisten Tools und Datenbanken verwenden sie standardmäßig
# - **Skala**: Mathematisch von -1.0 (entgegengesetzt) bis 1.0 (identisch)
#   - Für Text-Embeddings moderner Modelle: typischerweise 0.0 (unverwandt) bis 1.0
#   - Negative Werte sind bei Text-Embeddings selten, aber mathematisch möglich
# - **Effizienz**: Berechnung über Skalarprodukt (schnell!)
# - **Robustheit**: Funktioniert auch mit nicht-normalisierten Modellen
