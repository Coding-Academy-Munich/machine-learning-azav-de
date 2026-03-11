# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Semantische Suche mit Embeddings</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %%
import os
import dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from vector_embeddings_plots import plot_similarity_matrix

# %%
dotenv.load_dotenv()

# %%
texts = [
    "Machine Learning ist ein Teilbereich der KI",
    "ML ist eine wichtige KI-Methode",
    "Deep Learning nutzt neuronale Netze",
    "Katzen sind Haustiere",
    "Machine Learning is a subset of AI",
]

# %%
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

# %%
text_embeddings = embeddings.embed_documents(texts)

# %% [markdown]
#
# ## Ähnlichkeitsmatrix berechnen

# %%
similarities = cosine_similarity(text_embeddings)

# %%
plot_similarity_matrix(similarities, texts)

# %% [markdown]
#
# ## Was sehen wir?
#
# - Text 1 und 2 (ML und KI-Methode): **hohe Ähnlichkeit** (~0.8)
#   - Fast dieselbe Aussage, nur anders formuliert
# - Text 3 (Deep Learning) und 1–2: **mittlere Ähnlichkeit** (~0.5)
#   - Verwandtes KI-Thema, aber anderer Fokus
# - Text 4 (Katzen): **niedrige Ähnlichkeit** zu allen (~0.2)
#   - Hat nichts mit Technologie zu tun
#
# **Das ist die Stärke der semantischen Suche!**

# %% [markdown]
#
# ## Semantische Suche in Aktion

# %%
query = "Was sind neuronale Netze?"

# %% [markdown]
#
# Wir gehen folgendermaßen vor:
# 1. Berechnen des Embeddings der Anfrage mit `embeddings.embed_query(query)`
# 2. Berechnen der Kosinus-Ähnlichkeit zwischen Anfrage-Embedding und
#    Text-Embeddings
# 3. Nach Ähnlichkeit sortieren

# %%

# %%

# %%

# %%

# %%

# %%

# %%
my_values = [3, 1, 4, 2]
my_numbers = ["three", "one", "four", "two"]

# %%
my_sorted_indices = np.argsort(my_values)[::-1]


# %%

# %%

# %%

# %%
def print_by_similarity(similarities, texts):
    sorted_indices = np.argsort(similarities)[::-1]
    print("Texte sortiert nach Ähnlichkeit:")
    for i in sorted_indices:
        print(f"  [{similarities[i]:.3f}] {texts[i]}")

# %%

# %% [markdown]
#
# ## Warum nicht einfach Keyword-Suche?
#
# - **Keyword-Suche**: Sucht nach exakten Wörtern im Text
# - **Problem**:
#   - Findet nur exakte Übereinstimmungen
#   - Versteht keine Synonyme, keine Übersetzungen, keine verwandten Konzepte

# %%
def keyword_search(query, texts):
    return [t for t in texts if query.lower() in t.lower()]

# %% [markdown]
#
# ## Vergleich: Keyword vs. Semantische Suche

# %%
query1 = "neuronale Netze"
print(f"Suche: '{query1}'")
print(f"\nKeyword-Ergebnisse:")
kw_results = keyword_search(query1, texts)
if kw_results:
    for r in kw_results:
        print(f"  - {r}")
else:
    print("  (keine Treffer)")

# %%
q_emb = [embeddings.embed_query(query1)]
sims = cosine_similarity(q_emb, text_embeddings)[0]
top_indices = np.argsort(sims)[::-1][:2]

# %%
print(f"\nSemantische Ergebnisse:")
for idx in top_indices:
    print(f"  [{sims[idx]:.3f}] {texts[idx]}")

# %% [markdown]
#
# ## Keyword-Suche versagt bei anderen Sprachen

# %%

# %%
q_emb2 = [embeddings.embed_query(query2)]
sims2 = cosine_similarity(q_emb2, text_embeddings)[0]
top_indices2 = np.argsort(sims2)[::-1][:2]

# %%
print(f"\nSemantische Ergebnisse:")
for idx in top_indices2:
    print(f"  [{sims2[idx]:.3f}] {texts[idx]}")

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
# 2. ???
# 3. Bei Anfrage: Anfrage embedden
# 4. Ähnlichste Chunks finden (Kosinus-Ähnlichkeit)
# 5. Als Kontext an LLM geben

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
