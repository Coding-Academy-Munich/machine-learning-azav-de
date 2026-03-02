# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Vector Embeddings für RAG</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
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
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from langchain_openai import OpenAIEmbeddings

# %%
dotenv.load_dotenv()

# %% [markdown]
#
# ## Text-Embeddings
#
# - Nicht nur Wörter, sondern **ganze Texte** einbetten
# - Sätze, Absätze, Dokumente → Vektoren
# - **Dimension**: Meist 384, 768, oder 1536
#   - Mehr Dimensionen = mehr Nuancen, aber auch teurer
# - **Modelle**:
#   - OpenAI: text-embedding-3-small (1536 Dimensionen)
#   - HuggingFace: verschiedene Open-Source-Modelle
#   - Qdrant: unterstützt verschiedene Modelle

# %% [markdown]
#
# ## Embeddings mit LangChain erstellen

# %%
# TODO: Create an OpenAIEmbeddings model using OpenRouter
# embeddings = OpenAIEmbeddings(...)

# %%
texts = [
    "Machine Learning ist ein Teilbereich der KI",
    "Deep Learning nutzt neuronale Netze",
    "Python ist eine Programmiersprache",
    "Katzen sind Haustiere"
]

# %%
# TODO: Create embeddings for all texts
# text_embeddings = ...

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
#     <li>Misst $|\vec a - \vec b|$ (Spitze zu Spitze)</li>
#     <li>Hängt von <b>Richtung und Länge</b> der Vektoren ab</li>
#   </ul>
#   <li><b>Kosinus-Ähnlichkeit</b> (links): Winkel zwischen den Pfeilen</li>
#   <ul>
#     <li>Misst nur die <b>Richtung</b>, ignoriert die Länge</li>
#     <li>$\vec a$ und $\vec b$: kleiner Winkel → hohe Ähnlichkeit</li>
#   </ul>
# </ul>
# </div>

# %% [markdown]
#
# ## Beispiel: 2D-Vektoren
#
# - Betrachten wir drei einfache 2D-Vektoren (wie in den Bildern)
# - $\vec a$ und $\vec b$: gleiche Richtung, aber $\vec b$ ist kürzer
# - $\vec c$: zeigt in eine ganz andere Richtung

# %%
a = np.array([[4, 3]])
b = np.array([[2, 1.5]])
c = np.array([[1, -3]])

# %%

# %%

# %%

# %%

# %%
vectors = {"a": a[0], "b": b[0], "c": c[0]}
colors = {"a": "red", "b": "black", "c": "darkblue"}
fig, ax = plt.subplots(figsize=(6, 5))
for name, v in vectors.items():
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=colors[name], lw=2))
    ax.text(v[0] + 0.15, v[1] + 0.15, name, fontsize=14, color=colors[name])
ax.set_xlim(-1, 5.5)
ax.set_ylim(-4, 4.5)
ax.set_aspect("equal")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_title("2D-Vektoren / 2D Vectors")
plt.tight_layout()
plt.show()

# %% [markdown]
#
# ## Ergebnis: 2D-Beispiel
#
# - $\vec a$ und $\vec b$: **gleiche Richtung** (Kosinus-Ähnlichkeit = 1.0)
#   - Aber: Euklidische Distanz = 2.5 (wegen unterschiedlicher Länge)
# - $\vec a$ und $\vec c$: **verschiedene Richtungen** (Kosinus-Ähnlichkeit ≈ −0.45)
#   - Euklidische Distanz = 6.7 (weit entfernt)
# - Kosinus-Ähnlichkeit ignoriert die Länge und misst nur die Richtung

# %% [markdown]
#
# ## Moderne Embedding-Modelle normalisieren!
#
# - Modelle wie `text-embedding-3-small` geben **normalisierte Vektoren** zurück
# - Alle Vektoren haben die **gleiche Länge** (Norm = 1), egal wie lang der Text ist
# - Für normalisierte Vektoren:
#   - Euklidische Distanz und Kosinus-Ähnlichkeit ergeben **dieselbe Rangfolge**
#   - $d_{\text{euklid}} = \sqrt{2 \cdot (1 - \text{cos\_sim})}$
# - Der Unterschied aus dem 2D-Beispiel tritt bei Embeddings also **nicht** auf!

# %% [markdown]
#
# ## Warum trotzdem Kosinus-Ähnlichkeit?
#
# - **Konvention**: Die meisten Tools und Datenbanken verwenden sie standardmäßig
# - **Intuitive Skala**: 0.0 (völlig verschieden) bis 1.0 (identisch)
# - **Effizienz**: Berechnung über Skalarprodukt (schnell!)
# - **Robustheit**: Funktioniert auch mit nicht-normalisierten Modellen
#
# In der Praxis: Qdrant, Pinecone etc. nutzen Kosinus-Ähnlichkeit als Standard

# %% [markdown]
#
# ## Ähnlichkeitsmatrix berechnen

# %%
similarities = cosine_similarity(text_embeddings)

# %%

# %% [markdown]
#
# ## Was sehen wir?
#
# - Text 1 und 2 (ML und Deep Learning): **hohe Ähnlichkeit** (~0.8)
#   - Beide handeln von KI-Methoden
# - Text 3 (Python) und 1–2: **mittlere Ähnlichkeit** (~0.5)
#   - Python wird oft für ML genutzt — das "weiß" das Modell!
# - Text 4 (Katzen): **niedrige Ähnlichkeit** zu allen (~0.2–0.3)
#   - Hat nichts mit Technologie zu tun
#
# **Das ist die Stärke der semantischen Suche!**

# %% [markdown]
#
# ## Semantische Suche in Aktion

# %%
query = "Was sind neuronale Netze?"


# %%
# TODO: Embed the query and find the most similar texts
# 1. Embed the query with embeddings.embed_query(query)
# 2. Calculate cosine similarity
# 3. Sort by similarity

# %%

# %% [markdown]
#
# ## Warum nicht einfach Keyword-Suche?
#
# - **Keyword-Suche**: Sucht nach exakten Wörtern im Text
# - **Problem**: Findet nur exakte Übereinstimmungen
# - Versteht keine Synonyme, keine Übersetzungen, keine verwandten Konzepte

# %%
def keyword_search(query, texts):
    """Simple keyword search: returns texts containing the query."""
    return [t for t in texts if query.lower() in t.lower()]

# %% [markdown]
#
# ## Vergleich: Keyword vs. Semantische Suche

# %%

# %%

# %% [markdown]
#
# ## Keyword-Suche versagt bei anderen Sprachen

# %%

# %%

# %% [markdown]
#
# ## Semantische Suche versteht Bedeutung
#
# | | Keyword-Suche | Semantische Suche |
# |---|---|---|
# | **Exakte Wörter** | ✅ Findet sie | ✅ Findet sie |
# | **Synonyme** | ❌ "KI" ≠ "AI" | ✅ Versteht Bedeutung |
# | **Andere Sprache** | ❌ "AI" ≠ "KI" | ✅ Sprachübergreifend |
# | **Verwandte Konzepte** | ❌ Kein Zusammenhang | ✅ Erkennt Verwandtschaft |

# %% [markdown]
#
# ## Für RAG bedeutet das:
#
# 1. Alle Dokument-Chunks embedden
# 2. In Vektor-Datenbank speichern
# 3. Bei Anfrage: Anfrage embedden
# 4. Ähnlichste Chunks finden (Kosinus-Ähnlichkeit)
# 5. Als Kontext an LLM geben
#
# **Qdrant** macht Schritte 2–4 automatisch!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Text-Embeddings**: Ganze Texte als Vektoren darstellen
# - **Kosinus-Ähnlichkeit**: Standard-Metrik für Text-Ähnlichkeit
#   - Intuitive Skala: 0.0 (verschieden) bis 1.0 (identisch)
# - **Semantische Suche**: Versteht Bedeutung, nicht nur exakte Wörter
#   - Findet Synonyme, verwandte Konzepte, andere Sprachen
# - **LangChain**: Einfache Embedding-Erstellung mit wenigen Zeilen
#
# **Nächster Schritt**: Qdrant für effiziente Vektor-Speicherung!
