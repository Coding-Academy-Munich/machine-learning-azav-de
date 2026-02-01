# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Text-Klassifikation</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was ist Text-Klassifikation?
#
# - **Input**: Ein Text
# - **Output**: Eine Kategorie
# - Beispiele:
#   - Spam-Erkennung (Spam / Kein Spam)
#   - Sentiment-Analyse (Positiv / Negativ / Neutral)
#   - Themen-Klassifikation (Sport / Politik / Technologie)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
sns.set_theme(style="darkgrid")

# %% [markdown]
#
# ## Beispiel-Datensatz
#
# - Einfache Sentiment-Analyse
# - Texte über Produkte
# - Labels: Positiv (1) oder Negativ (0)

# %%
# Sample dataset
texts = [
    "This product is amazing! I love it!",
    "Terrible quality, waste of money",
    "Best purchase ever, highly recommend",
    "Worst product I've ever bought",
    "Great value for money",
    "Do not buy this, very disappointed",
    "Excellent product, works perfectly",
    "Poor quality, broke after one day",
    "Love it! Exactly what I needed",
    "Not worth the price, very bad",
    "Outstanding quality and fast delivery",
    "Horrible experience, would not recommend",
    "Perfect! Better than expected",
    "Cheap and breaks easily",
    "Fantastic product, very happy",
    "Waste of time and money"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative

# %%

# %% [markdown]
#
# ## Bag of Words
#
# - **Bag of Words** (BoW) = Einfachste Methode
# - Zähle, wie oft jedes Wort vorkommt
# - Ignoriert Reihenfolge komplett
# - Jeder Text → Vektor von Wort-Zählungen

# %%
# Create Bag of Words representation
vectorizer = CountVectorizer(lowercase=True, max_features=20)
X_bow = vectorizer.fit_transform(texts)


# %%

# %%

# %%
def plot_bow_example(text, vectorizer, bow_vector):
    """Visualize Bag of Words for one text"""
    words = vectorizer.get_feature_names_out()
    counts = bow_vector.toarray()[0]

    # Only show non-zero counts
    non_zero = counts > 0
    words_shown = words[non_zero]
    counts_shown = counts[non_zero]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(words_shown)), counts_shown, alpha=0.7)
    plt.xticks(range(len(words_shown)), words_shown, rotation=45, ha='right')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Bag of Words for: "{text}"', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## TF-IDF: Wichtigere Wörter
#
# - **TF-IDF** = Term Frequency - Inverse Document Frequency
# - Nicht nur zählen, sondern **gewichten**
# - Häufige Wörter (the, is, a) bekommen weniger Gewicht
# - Seltene, spezifische Wörter bekommen mehr Gewicht

# %%
# Create TF-IDF representation
tfidf_vectorizer = TfidfVectorizer(lowercase=True, max_features=20)
X_tfidf = tfidf_vectorizer.fit_transform(texts)


# %%

# %%
def plot_tfidf_comparison(text, vectorizer, tfidf_vector):
    """Visualize TF-IDF weights for one text"""
    words = vectorizer.get_feature_names_out()
    weights = tfidf_vector.toarray()[0]

    # Only show non-zero weights
    non_zero = weights > 0
    words_shown = words[non_zero]
    weights_shown = weights[non_zero]

    # Sort by weight
    sorted_indices = np.argsort(weights_shown)[::-1]
    words_shown = words_shown[sorted_indices]
    weights_shown = weights_shown[sorted_indices]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(words_shown)), weights_shown, alpha=0.7, color='lightcoral')
    plt.xticks(range(len(words_shown)), words_shown, rotation=45, ha='right')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('TF-IDF Weight', fontsize=12)
    plt.title(f'TF-IDF for: "{text}"', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Modell trainieren
#
# - Jetzt haben wir Vektoren für jeden Text
# - Können wir ein normales ML-Modell verwenden
# - Probieren wir Logistic Regression

# %%
# Convert to numpy array
X = X_tfidf.toarray()
y = np.array(labels)

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%

# %%
# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# %%

# %% [markdown]
#
# ## Vorhersagen machen

# %%
# New texts to classify
new_texts = [
    "This is the best thing ever!",
    "Absolutely terrible, do not recommend",
    "Pretty good, I like it"
]

# %%
# Transform and predict
new_X = tfidf_vectorizer.transform(new_texts)
predictions = model.predict(new_X)
probabilities = model.predict_proba(new_X)


# %%

# %%
def plot_predictions(texts, predictions, probabilities):
    """Visualize predictions and confidence"""
    fig, axes = plt.subplots(len(texts), 1, figsize=(12, 4*len(texts)))

    if len(texts) == 1:
        axes = [axes]

    for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
        sentiment = "Positive" if pred == 1 else "Negative"
        color = 'green' if pred == 1 else 'red'

        # Show probabilities
        axes[i].barh(['Negative', 'Positive'], prob, color=['red', 'green'], alpha=0.6)
        axes[i].set_xlim(0, 1)
        axes[i].set_xlabel('Probability', fontsize=11)
        axes[i].set_title(f'"{text}"\nPrediction: {sentiment}',
                         fontsize=12, fontweight='bold', color=color)
        axes[i].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Wichtigste Wörter für jede Klasse

# %%
# Get feature importance (coefficients)
feature_names = tfidf_vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Top words for positive class
top_positive_indices = np.argsort(coefficients)[-5:]
top_positive_words = [(feature_names[i], coefficients[i]) for i in top_positive_indices]

# Top words for negative class
top_negative_indices = np.argsort(coefficients)[:5]
top_negative_words = [(feature_names[i], coefficients[i]) for i in top_negative_indices]


# %%

# %%
def plot_feature_importance(positive_words, negative_words):
    """Plot most important features for each class"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Positive words
    words_pos = [w for w, _ in reversed(positive_words)]
    coefs_pos = [c for _, c in reversed(positive_words)]
    ax1.barh(words_pos, coefs_pos, color='green', alpha=0.6)
    ax1.set_xlabel('Coefficient', fontsize=12)
    ax1.set_title('Top Words for POSITIVE', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Negative words
    words_neg = [w for w, _ in negative_words]
    coefs_neg = [c for _, c in negative_words]
    ax2.barh(words_neg, coefs_neg, color='red', alpha=0.6)
    ax2.set_xlabel('Coefficient', fontsize=12)
    ax2.set_title('Top Words for NEGATIVE', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Confusion Matrix

# %%
def plot_confusion_matrix_text(y_true, y_pred):
    """Plot confusion matrix for text classification"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%

# %% [markdown]
#
# ## Limitierungen von Bag of Words
#
# - **Ignoriert Reihenfolge**: "not good" vs "good" haben ähnliche Vektoren
# - **Ignoriert Kontext**: Keine Information über Satzbau
# - **Großes Vokabular**: Viele Features
# - Für einfache Aufgaben: Gut genug
# - Für komplexe Aufgaben: Neuronale Netze besser

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Text-Klassifikation**: Text → Kategorie
# - **Bag of Words**: Zähle Wörter, ignoriere Reihenfolge
# - **TF-IDF**: Gewichte wichtige Wörter höher
# - **Standard ML-Modelle** funktionieren mit diesen Vektoren
# - **Interpretierbar**: Können sehen, welche Wörter wichtig sind
# - Aber: Für komplexe Aufgaben brauchen wir mehr

# %% [markdown]
#
# ## Nächste Schritte
#
# - Sequenzielle Modelle (RNNs, LSTMs)
# - Transformers und Attention
# - Large Language Models (LLMs)
# - Das sind die modernen Ansätze!

# %%
