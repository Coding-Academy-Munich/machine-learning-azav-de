# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Deep Dive: Retrieval-Metriken</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Information Retrieval Metriken
#
# Die Metriken, die RAGAS und LangSmith verwenden, basieren auf
# klassischen **Information Retrieval** Metriken.
#
# In diesem Deep Dive schauen wir uns die Mathematik dahinter an:
# - Precision@k
# - Recall@k
# - Mean Reciprocal Rank (MRR)
# - NDCG (Normalized Discounted Cumulative Gain)

# %%
import numpy as np

# %% [markdown]
#
# ## Precision@k
#
# **"Wie viele der Top-k Ergebnisse sind relevant?"**
#
# $$\text{Precision@k} = \frac{|\text{relevante Docs in Top-k}|}{k}$$
#
# - $k$ = Anzahl der abgerufenen Dokumente
# - Wertebereich: 0.0 bis 1.0
# - Höher ist besser

# %% [markdown]
#
# ## Precision@k: Beispiel
#
# Unser Retriever gibt 4 Dokumente zurück.
# Relevante Dokumente sind mit R markiert, irrelevante mit X:
#
# ```
# Position:  1    2    3    4
# Relevanz:  R    X    R    X
# ```
#
# $$\text{Precision@4} = \frac{2}{4} = 0.5$$
#
# Nur die Hälfte der Ergebnisse ist relevant!

# %%
def precision_at_k(retrieved_relevant, k):
    """Calculate Precision@k.

    Args:
        retrieved_relevant: list of booleans, True if document at position i is relevant
        k: number of top results to consider
    """
    top_k = retrieved_relevant[:k]
    return sum(top_k) / k

# %%

# %%

# %% [markdown]
#
# ## Recall@k
#
# **"Wie viele aller relevanten Dokumente haben wir gefunden?"**
#
# $$\text{Recall@k} = \frac{|\text{relevante Docs in Top-k}|}{|\text{alle relevanten Docs}|}$$
#
# - Braucht die Gesamtzahl relevanter Dokumente
# - Wertebereich: 0.0 bis 1.0
# - Höher ist besser

# %% [markdown]
#
# ## Recall@k: Beispiel
#
# Insgesamt gibt es 3 relevante Dokumente im Corpus.
# Unser Retriever findet 2 davon in den Top-4:
#
# ```
# Top-4:          R    X    R    X
# Alle relevanten: R₁, R₂, R₃ (3 insgesamt)
# Gefunden:        R₁,     R₃           (2 gefunden)
# ```
#
# $$\text{Recall@4} = \frac{2}{3} = 0.67$$
#
# Ein relevantes Dokument wurde nicht gefunden!

# %%
def recall_at_k(retrieved_relevant, k, total_relevant):
    """Calculate Recall@k.

    Args:
        retrieved_relevant: list of booleans
        k: number of top results to consider
        total_relevant: total number of relevant documents in corpus
    """
    top_k = retrieved_relevant[:k]
    return sum(top_k) / total_relevant

# %%

# %% [markdown]
#
# ## Precision vs. Recall: Der Tradeoff
#
# - **Mehr Dokumente abrufen** (höheres k):
#   - Recall steigt (mehr relevante gefunden)
#   - Precision sinkt (mehr irrelevante dabei)
# - **Weniger Dokumente abrufen** (niedrigeres k):
#   - Precision steigt (fokussierter)
#   - Recall sinkt (könnte relevante verpassen)
#
# **Für RAG**: Recall ist oft wichtiger als Precision

# %% [markdown]
#
# ## Mean Reciprocal Rank (MRR)
#
# **"Wie weit oben ist das erste relevante Ergebnis?"**
#
# $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
#
# - $|Q|$ = Anzahl der Anfragen
# - $\text{rank}_i$ = Position des ersten relevanten Ergebnisses für Anfrage $i$
# - Wertebereich: 0.0 bis 1.0
# - Höher = relevantes Ergebnis weiter oben

# %% [markdown]
#
# ## MRR: Beispiel
#
# Drei Anfragen mit ihren Ergebnissen (R=relevant, X=irrelevant):
#
# | Anfrage | Ergebnisse | Erster relevanter Rang | Reciprocal Rank |
# |---------|-----------|----------------------|-----------------|
# | Q1 | **R** X X X | 1 | 1/1 = 1.0 |
# | Q2 | X **R** X X | 2 | 1/2 = 0.5 |
# | Q3 | X X X **R** | 4 | 1/4 = 0.25 |
#
# $$\text{MRR} = \frac{1.0 + 0.5 + 0.25}{3} = 0.583$$

# %%
def reciprocal_rank(retrieved_relevant):
    """Calculate Reciprocal Rank for a single query."""
    for i, is_relevant in enumerate(retrieved_relevant, 1):
        if is_relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(all_queries_relevant):
    """Calculate MRR across multiple queries."""
    rrs = [reciprocal_rank(q) for q in all_queries_relevant]
    return np.mean(rrs)

# %%

# %% [markdown]
#
# ## NDCG (Normalized Discounted Cumulative Gain)
#
# **"Wie gut ist die Reihenfolge der Ergebnisse?"**
#
# - Berücksichtigt **abgestufte Relevanz** (nicht nur relevant/irrelevant)
# - Bessere Ergebnisse weiter oben sind **wichtiger**
# - Normalisiert auf den bestmöglichen Wert

# %% [markdown]
#
# ## NDCG Schritt für Schritt
#
# **Schritt 1: DCG (Discounted Cumulative Gain)**
#
# $$\text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}$$
#
# - $\text{rel}_i$ = Relevanz des Dokuments an Position $i$
# - $\log_2(i+1)$ = Discount-Faktor (höhere Positionen → weniger Gewicht)
#
# **Schritt 2: Ideal DCG (IDCG)**
# - DCG der bestmöglichen Sortierung
#
# **Schritt 3: NDCG**
#
# $$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

# %% [markdown]
#
# ## NDCG: Beispiel
#
# Relevanz-Scores: 3=sehr relevant, 2=relevant, 1=teilweise, 0=irrelevant
#
# **Aktuelle Reihenfolge**: [3, 0, 2, 1]
#
# | Position $i$ | $\text{rel}_i$ | $\log_2(i+1)$ | $\frac{\text{rel}_i}{\log_2(i+1)}$ |
# |:-:|:-:|:-:|:-:|
# | 1 | 3 | 1.00 | 3.00 |
# | 2 | 0 | 1.58 | 0.00 |
# | 3 | 2 | 2.00 | 1.00 |
# | 4 | 1 | 2.32 | 0.43 |
#
# $\text{DCG@4} = 3.00 + 0.00 + 1.00 + 0.43 = 4.43$

# %% [markdown]
#
# **Ideale Reihenfolge**: [3, 2, 1, 0] (absteigend nach Relevanz)
#
# | Position $i$ | $\text{rel}_i$ | $\log_2(i+1)$ | $\frac{\text{rel}_i}{\log_2(i+1)}$ |
# |:-:|:-:|:-:|:-:|
# | 1 | 3 | 1.00 | 3.00 |
# | 2 | 2 | 1.58 | 1.26 |
# | 3 | 1 | 2.00 | 0.50 |
# | 4 | 0 | 2.32 | 0.00 |
#
# $\text{IDCG@4} = 3.00 + 1.26 + 0.50 + 0.00 = 4.76$
#
# $$\text{NDCG@4} = \frac{4.43}{4.76} = 0.93$$

# %%
def dcg_at_k(relevances, k):
    """Calculate DCG@k."""
    relevances = np.array(relevances[:k])
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances, k):
    """Calculate NDCG@k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

# %%

# %%

# %%

# %% [markdown]
#
# ## Vergleich: Gute vs. schlechte Reihenfolge

# %%
good_ranking = [3, 2, 1, 0]
bad_ranking = [0, 1, 2, 3]
mixed_ranking = [3, 0, 2, 1]

# %%

# %%

# %%

# %% [markdown]
#
# ## Welche Metrik wofür?
#
# | Metrik | Stärke | Schwäche |
# |--------|--------|---------|
# | **Precision@k** | Einfach, intuitiv | Ignoriert Reihenfolge |
# | **Recall@k** | Misst Vollständigkeit | Braucht Ground Truth |
# | **MRR** | Fokus auf erstes Ergebnis | Ignoriert Rest |
# | **NDCG** | Berücksichtigt Reihenfolge und Abstufung | Komplexer |
#
# **Für RAG empfohlen**: Context Recall (≈ Recall@k) + Faithfulness

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Precision@k**: Anteil relevanter Ergebnisse in Top-k
# - **Recall@k**: Anteil gefundener relevanter Ergebnisse
# - **MRR**: Wie schnell finden wir das erste relevante Ergebnis?
# - **NDCG**: Qualität der Gesamtreihenfolge mit Relevanz-Abstufung
# - RAGAS `Context Recall` basiert auf diesen klassischen Metriken
# - Für RAG-Systeme ist Recall besonders wichtig
