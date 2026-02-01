# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Word Embeddings</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Das Problem mit One-Hot Encoding
#
# - Erinnerung: One-Hot ist ineffizient und bedeutungslos
# - 10.000 Wörter = 10.000-dimensionaler Vektor
# - "king" und "queen" sind genau so verschieden wie "king" und "apple"
# - Wir brauchen **bessere** Wort-Repräsentationen

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Die Lösung: Word Embeddings
#
# - **Embedding** = Dichte Vektor-Repräsentation
# - Statt 10.000 Dimensionen: Nur 50-300
# - **Wichtig**: Ähnliche Wörter haben ähnliche Vektoren!
# - Lernt **Bedeutung** von Wörtern

# %% [markdown]
#
# ## One-Hot vs. Embedding

# %%
def plot_onehot_vs_embedding():
    """Compare one-hot and embedding representations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # One-hot (sparse)
    onehot = np.zeros(10)
    onehot[3] = 1
    ax1.bar(range(10), onehot, alpha=0.7, color='lightblue')
    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('One-Hot Vector\n(10,000 dimensions, mostly zeros)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.2)
    ax1.grid(True, alpha=0.3, axis='y')

    # Embedding (dense)
    embedding = np.array([0.2, -0.5, 0.8, 0.1, -0.3])
    ax2.bar(range(5), embedding, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Embedding Vector\n(50-300 dimensions, dense)', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Beispiel: Word2Vec (konzeptionell)
#
# - Eine berühmte Methode für Word Embeddings
# - Idee: "You shall know a word by the company it keeps"
# - Wörter, die in ähnlichen Kontexten vorkommen, sind ähnlich
# - Das Netz lernt die Embeddings automatisch

# %% [markdown]
#
# ## Beispiel-Embeddings erstellen
#
# - Wir erstellen vereinfachte Embeddings für Demo-Zwecke
# - In der Praxis: Werden durch Training gelernt
# - Oder: Vortrainierte Embeddings verwenden (Word2Vec, GloVe, etc.)

# %%
# Create example embeddings (simplified, hand-crafted for demonstration)
# Dimensions could represent: [formality, positivity, concreteness, ...]
word_embeddings = {
    'king': np.array([0.9, 0.3, 0.8]),
    'queen': np.array([0.9, 0.3, 0.7]),
    'man': np.array([0.5, 0.0, 0.9]),
    'woman': np.array([0.5, 0.0, 0.8]),
    'prince': np.array([0.8, 0.4, 0.85]),
    'princess': np.array([0.8, 0.4, 0.75]),
    'apple': np.array([-0.5, 0.2, 1.0]),
    'orange': np.array([-0.5, 0.3, 1.0]),
    'computer': np.array([0.3, -0.1, 1.0]),
    'laptop': np.array([0.3, -0.1, 0.95]),
}


# %%

# %% [markdown]
#
# ## Ähnlichkeit zwischen Wörtern
#
# - Wie misst man Ähnlichkeit zwischen Vektoren?
# - **Cosine Similarity**: Winkel zwischen Vektoren
# - Wert zwischen -1 und 1
# - 1 = Identisch, 0 = Orthogonal, -1 = Gegenteilig

# %%
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# %%

# %% [markdown]
#
# ## Visualisierung in 2D
#
# - Unsere Embeddings haben 3 Dimensionen
# - Schwer zu visualisieren!
# - Lösung: **PCA** (Principal Component Analysis)
# - Reduziert auf 2D für Plotting

# %%
# Prepare data for PCA
words = list(word_embeddings.keys())
embeddings_matrix = np.array([word_embeddings[word] for word in words])

# %%
# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_matrix)

# %%
def plot_embeddings_2d(words, embeddings_2d):
    """Plot word embeddings in 2D space"""
    plt.figure(figsize=(12, 10))

    # Plot points
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=200, alpha=0.6)

    # Add labels
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=12, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')

    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title('Word Embeddings Visualization (PCA 2D)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Was sehen wir?
#
# - **Ähnliche Wörter liegen nahe beieinander!**
# - Königsfamilie (king, queen, prince, princess) gruppiert
# - Geschlecht (man, woman) gruppiert
# - Früchte (apple, orange) gruppiert
# - Technologie (computer, laptop) gruppiert

# %% [markdown]
#
# ## Analogien: Die berühmte Gleichung
#
# - **king - man + woman ≈ queen**
# - Vektor-Arithmetik funktioniert!
# - "Königlichkeit" + "Weiblichkeit" = "Königin"
# - Das ist die Magie von Embeddings

# %%
# Compute analogy
analogy_vector = (word_embeddings['king'] -
                 word_embeddings['man'] +
                 word_embeddings['woman'])

# Find closest word
def find_closest_word(target_vector, word_embeddings, exclude=[]):
    """Find word with embedding closest to target vector"""
    best_similarity = -999
    best_word = None

    for word, embedding in word_embeddings.items():
        if word in exclude:
            continue
        similarity = cosine_similarity(target_vector, embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_word = word

    return best_word, best_similarity

# %%

# %%
def plot_analogy_visualization():
    """Visualize word analogy as vectors"""
    # Get 2D positions
    king_idx = words.index('king')
    man_idx = words.index('man')
    woman_idx = words.index('woman')
    queen_idx = words.index('queen')

    king_pos = embeddings_2d[king_idx]
    man_pos = embeddings_2d[man_idx]
    woman_pos = embeddings_2d[woman_idx]
    queen_pos = embeddings_2d[queen_idx]

    plt.figure(figsize=(12, 10))

    # Plot all points
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.3, color='gray')

    # Highlight relevant points
    plt.scatter(*king_pos, s=300, color='blue', label='king', zorder=5)
    plt.scatter(*man_pos, s=300, color='red', label='man', zorder=5)
    plt.scatter(*woman_pos, s=300, color='green', label='woman', zorder=5)
    plt.scatter(*queen_pos, s=300, color='purple', label='queen', zorder=5)

    # Draw arrows
    # king - man
    # plt.arrow(king_pos[0], king_pos[1],
    #          -(king_pos[0] - man_pos[0]) * 0.9, -(king_pos[1] - man_pos[1]) * 0.9,
    #          head_width=0.01, head_length=0.01, fc='red', ec='red', linewidth=2,
    #          length_includes_head=True, label='- man')

    # + woman
    result_pos = king_pos - (king_pos - man_pos)
    # plt.arrow(result_pos[0], result_pos[1],
    #          (woman_pos[0] - man_pos[0]) * 0.9, (woman_pos[1] - man_pos[1]) * 0.9,
    #          head_width=0.01, head_length=0.01, fc='green', ec='green', linewidth=2,
    #          length_includes_head=True, label='+ woman')

    # Labels
    for i, word in enumerate(words):
        if word in ['king', 'man', 'woman', 'queen']:
            plt.annotate(word, embeddings_2d[i], fontsize=12, fontweight='bold',
                        xytext=(8, 8), textcoords='offset points')

    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title('Word Analogy: king - man + woman ≈ queen', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Dimensionen haben Bedeutung
#
# - Jede Dimension kodiert ein **Konzept**
# - Zum Beispiel:
#   - Dimension 1: Geschlecht (männlich/weiblich)
#   - Dimension 2: Lebewesen vs. Objekt
#   - Dimension 3: Konkret vs. Abstrakt
# - In der Praxis: Nicht so klar interpretierbar

# %% [markdown]
#
# ## Embeddings in der Praxis
#
# - **Vortrainierte Embeddings**:
#   - Word2Vec (Google)
#   - GloVe (Stanford)
#   - FastText (Facebook)
# - Auf riesigen Text-Korpora trainiert
# - Kann man direkt verwenden oder fine-tunen

# %% [markdown]
#
# ## Von Wort-Embeddings zu...
#
# - Wort-Embeddings sind **statisch**
# - Jedes Wort hat **einen** Vektor
# - Problem: "bank" (Geldinstitut) vs. "bank" (Flussufer)
# - Lösung: **Kontextuelle Embeddings**
# - Jedes Wort bekommt einen Vektor **abhängig vom Kontext**
# - Das führt uns zu modernen LLMs!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Embeddings**: Dichte Vektor-Repräsentationen von Wörtern
# - **Effizienter**: 50-300 statt 10.000 Dimensionen
# - **Bedeutungsvoll**: Ähnliche Wörter → ähnliche Vektoren
# - **Analogien**: Vektor-Arithmetik funktioniert!
# - **Praktisch**: Vortrainierte Embeddings verfügbar
# - **Grundlage**: Für moderne NLP und LLMs

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - Wie verarbeitet man **Sequenzen** von Wörtern?
# - Einfache Modelle für Text-Klassifikation
# - Praktisches Beispiel mit echten Daten

# %%
