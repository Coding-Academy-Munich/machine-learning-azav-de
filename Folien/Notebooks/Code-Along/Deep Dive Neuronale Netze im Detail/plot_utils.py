import numpy as np
import matplotlib.pyplot as plt


def plot_text_to_ids(text, word_to_id):
    """Visualize conversion from text to IDs"""
    def simple_tokenize(text):
        """Split text into words (lowercase)"""
        return text.lower().split()

    def text_to_ids(text, word_to_id):
        """Convert text to sequence of word IDs"""
        tokens = simple_tokenize(text)
        return [word_to_id.get(word, -1) for word in tokens]

    tokens = simple_tokenize(text)
    ids = text_to_ids(text, word_to_id)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Tokens
    for i, token in enumerate(tokens):
        ax1.text(i, 0, token, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black'))
    ax1.set_xlim(-0.5, len(tokens) - 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_title('Tokens (Words)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # IDs
    for i, id_num in enumerate(ids):
        ax2.text(i, 0, str(id_num), ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black'))
        # Arrow
        ax2.arrow(i, 0.15, 0, -0.08, head_width=0.15, head_length=0.03,
                 fc='gray', ec='gray')
    ax2.set_xlim(-0.5, len(ids) - 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_title('Word IDs', fontsize=14, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def plot_word_frequencies(word_counts):
    """Plot word frequency distribution"""
    words = [word for word, _ in word_counts.most_common()]
    counts = [count for _, count in word_counts.most_common()]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(words)), counts, alpha=0.7)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Word Frequency Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_onehot_encoding(word, word_to_id, vocabulary):
    """Visualize one-hot encoding"""
    def word_to_onehot(word, word_to_id, vocab_size):
        """Convert word to one-hot vector"""
        vector = np.zeros(vocab_size)
        if word in word_to_id:
            vector[word_to_id[word]] = 1
        return vector

    vocab_size = len(vocabulary)
    onehot = word_to_onehot(word, word_to_id, vocab_size)

    plt.figure(figsize=(12, 4))
    plt.bar(range(vocab_size), onehot, alpha=0.7)
    plt.xticks(range(vocab_size), vocabulary, rotation=45, ha='right')
    plt.xlabel('Vocabulary', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'One-Hot Encoding for "{word}"', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_sequence_lengths(sequences):
    """Plot distribution of sequence lengths"""
    lengths = [len(seq) for seq in sequences]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(lengths)), lengths, alpha=0.7)
    plt.xlabel('Text Index', fontsize=12)
    plt.ylabel('Sequence Length (# words)', fontsize=12)
    plt.title('Sequence Length Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_padding_visualization(original_sequences, padded_sequences):
    """Visualize padding effect"""
    fig, axes = plt.subplots(len(original_sequences), 1, figsize=(12, 8))

    if len(original_sequences) == 1:
        axes = [axes]

    for i, (orig, padded) in enumerate(zip(original_sequences, padded_sequences)):
        # Original length
        axes[i].barh([0], [len(orig)], color='lightblue', label='Original' if i == 0 else '', edgecolor='black')
        # Padding
        axes[i].barh([0], [len(padded)], left=[len(orig)], color='lightgray',
                    label='Padding' if i == 0 else '', edgecolor='black')
        axes[i].set_xlim(0, len(padded))
        axes[i].set_ylim(-0.5, 0.5)
        axes[i].set_yticks([])
        axes[i].set_xlabel('Sequence Position', fontsize=10)
        axes[i].set_title(f'Sequence {i+1}', fontsize=11)
        axes[i].grid(True, alpha=0.3, axis='x')

    if len(original_sequences) > 0:
        axes[0].legend()

    plt.suptitle('Padding Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
