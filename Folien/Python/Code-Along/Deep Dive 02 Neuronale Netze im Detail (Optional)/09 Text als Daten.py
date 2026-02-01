# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Text als Daten</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Text und Machine Learning
#
# - Bisher: Zahlen und Bilder
# - Jetzt: **Text**
# - Text ist überall: E-Mails, Artikel, Chat-Nachrichten
# - Aber wie verarbeiten Computer Text?

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Das Problem
#
# - Machine Learning braucht **Zahlen**
# - Text besteht aus **Wörtern**
# - Wir müssen Text in Zahlen umwandeln
# - Aber wie?

# %% [markdown]
#
# ## Beispiel-Text

# %%
# Sample texts
texts = [
    "I love machine learning",
    "Machine learning is awesome",
    "Deep learning is a subset of machine learning",
    "I love deep learning too"
]


# %%

# %% [markdown]
#
# ## Schritt 1: Tokenisierung
#
# - **Tokenisierung** = Text in einzelne Teile zerlegen
# - Meist: Wörter (aber auch Zeichen, Silben möglich)
# - Einfachste Methode: Bei Leerzeichen trennen

# %%
# Simple tokenization
def simple_tokenize(text):
    """Split text into words (lowercase)"""
    return text.lower().split()

# %%

# %% [markdown]
#
# ## Alle Texte tokenisieren

# %%
# Tokenize all texts
all_tokens = [simple_tokenize(text) for text in texts]

# %%

# %% [markdown]
#
# ## Vokabular erstellen
#
# - Sammle alle **einzigartigen Wörter**
# - Das ist unser **Vokabular** (Vocabulary)
# - Jedes Wort bekommt eine **ID-Nummer**

# %%
# Build vocabulary
vocabulary = set()
for tokens in all_tokens:
    vocabulary.update(tokens)

vocabulary = sorted(list(vocabulary))

# %%

# %%
# Create word to ID mapping
word_to_id = {word: idx for idx, word in enumerate(vocabulary)}
id_to_word = {idx: word for word, idx in word_to_id.items()}


# %%

# %% [markdown]
#
# ## Text in Zahlen umwandeln

# %%
# Convert text to sequence of IDs
def text_to_ids(text, word_to_id):
    """Convert text to sequence of word IDs"""
    tokens = simple_tokenize(text)
    return [word_to_id.get(word, -1) for word in tokens]

# %%

# %% [markdown]
#
# ## Visualisierung: Text → IDs

# %%
from word_embeddings_plots import (plot_text_to_ids, plot_word_frequencies,
                        plot_onehot_encoding, plot_sequence_lengths,
                        plot_padding_visualization)

# %%

# %% [markdown]
#
# ## Häufigkeit von Wörtern

# %%
# Count word frequencies
all_words = []
for tokens in all_tokens:
    all_words.extend(tokens)

word_counts = Counter(all_words)


# %%

# %%

# %% [markdown]
#
# ## One-Hot Encoding
#
# - Eine Methode, Wörter als Vektoren darzustellen
# - Jedes Wort → Vektor der Länge Vokabular-Größe
# - Nur eine 1, Rest 0en
# - Position der 1 = Wort-ID

# %%
def word_to_onehot(word, word_to_id, vocab_size):
    """Convert word to one-hot vector"""
    vector = np.zeros(vocab_size)
    if word in word_to_id:
        vector[word_to_id[word]] = 1
    return vector

# %%

# %%

# %% [markdown]
#
# ## Problem mit One-Hot Encoding
#
# - **Sehr groß**: Bei 10.000 Wörtern → 10.000-dimensionaler Vektor!
# - **Sparsam**: Nur eine 1, Rest 0en (ineffizient)
# - **Keine Bedeutung**: "king" und "queen" sind genau so verschieden wie "king" und "apple"
# - Wir brauchen etwas Besseres!

# %% [markdown]
#
# ## Sequenzen sind wichtig
#
# - Bei Text ist die **Reihenfolge** wichtig!
# - "I love machine learning" ≠ "machine learning love I"
# - Im Gegensatz zu Bildern: Position der Pixel ist fest
# - Text ist **sequenziell**

# %% [markdown]
#
# ## Verschiedene Textlängen
#
# - Sätze haben unterschiedliche Längen
# - "Hi" vs "This is a very long sentence with many words"
# - Machine Learning Modelle brauchen oft feste Input-Größe
# - Lösungen: Padding, Truncation

# %%
# Show different sequence lengths
sequences = [text_to_ids(text, word_to_id) for text in texts]


# %%

# %%

# %% [markdown]
#
# ## Padding
#
# - **Padding** = Auffüllen kurzer Sequenzen
# - Meist mit 0 oder speziellem PAD-Token
# - Alle Sequenzen auf gleiche Länge bringen

# %%
def pad_sequences(sequences, max_length=None, pad_value=0):
    """Pad sequences to same length"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            # Pad at end
            padded_seq = seq + [pad_value] * (max_length - len(seq))
        else:
            # Truncate if too long
            padded_seq = seq[:max_length]
        padded.append(padded_seq)

    return np.array(padded)

# %%

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Text muss in Zahlen** umgewandelt werden
# - **Tokenisierung**: Text → Wörter
# - **Vokabular**: Alle einzigartigen Wörter mit IDs
# - **One-Hot Encoding**: Einfach, aber ineffizient und ohne Bedeutung
# - **Sequenzen**: Reihenfolge ist wichtig!
# - **Padding**: Gleiche Länge für alle Sequenzen

# %% [markdown]
#
# ## In der nächsten Lektion
#
# - **Word Embeddings**: Bessere Repräsentation von Wörtern
# - Wörter mit ähnlicher Bedeutung liegen nahe beieinander
# - Viel effizienter als One-Hot Encoding

# %%
