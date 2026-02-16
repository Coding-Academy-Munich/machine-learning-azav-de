# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Text-Chunking</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%

# %%
import re
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# %% [markdown]
#
# ## Chunking: Text aufteilen
#
# - **Problem**: Lange Dokumente passen nicht in LLM-Context
# - **Lösung**: In kleinere **Chunks** aufteilen
# - **Wichtig**: Chunks sollten
#   - Nicht zu klein (Kontext fehlt)
#   - Nicht zu groß (Context-Limit)
#   - Sinnvoll sein (nicht mitten im Satz!)

# %% [markdown]
#
# ## Beispiel-Dokument

# %%
with open("docs/ai-intro.txt", "r") as f:
    document = f.read()

# %%

# %% [markdown]
#
# ## LangChain Text Splitters
#
# - **CharacterTextSplitter**: Teilt bei einem bestimmten Zeichen (z.B. `\n`)
# - **RecursiveCharacterTextSplitter**: Schlauer — probiert nacheinander:
#   - Absätze (`\n\n`) → Zeilenumbrüche (`\n`) → Sätze (`. `) → Wörter (` `)
# - **Empfohlen**: RecursiveCharacterTextSplitter
#   - Erhält die Struktur des Textes besser

# %%
# TODO: Create a RecursiveCharacterTextSplitter
# with chunk_size=200, chunk_overlap=50

# %%

# %%

# %% [markdown]
#
# ## Chunk-Parameter verstehen
#
# - **chunk_size**: Maximale Größe eines Chunks (in Zeichen)
# - **chunk_overlap**: Überlappung zwischen aufeinanderfolgenden Chunks
#   - Warum? Kontext am Rand nicht verlieren!
# - **separators**: Wo wird geteilt? (Absatz → Zeile → Satz → Wort)
#
# **Typische Werte**:
# - chunk_size: 500–1000 für Dokumente
# - chunk_overlap: 50–200 (10–20% von chunk_size)

# %% [markdown]
#
# ## Überlappung visualisiert
#
# ```
# Chunk 1: [==============================]
# Chunk 2:                    [==============================]
#                             ^^^^^^^^^^^^
#                         Überlappungsbereich
# ```
#
# - Text am Ende von Chunk 1 erscheint auch am Anfang von Chunk 2
# - Sätze an der Grenze gehen nicht verloren
# - Besonders wichtig, wenn ein Satz genau an der Chunk-Grenze liegt

# %% [markdown]
#
# ## Überlappung im Code

# %%
splitter_no_overlap = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
splitter_with_overlap = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

# %%

# %%

# %%

# %% [markdown]
#
# ## Tokens vs. Zeichen
#
# - Erinnerung: LLMs zählen in **Tokens**, nicht Zeichen
# - **Faustregel**: 1 Token ≈ 4 Zeichen (für englischen Text)
# - Für deutschen Text etwas weniger (wegen längerer Wörter)
# - LangChain bietet auch einen **Token-basierten Splitter**

# %%
token_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=10)


# %%

# %% [markdown]
#
# ## Bereinigen + Chunken: Alles zusammen
#
# - In der Praxis kombinieren wir Bereinigung und Chunking
# - Wir verwenden unsere `clean_text`-Funktion aus dem vorherigen Abschnitt

# %%
def clean_text(text):
    """Clean text by removing HTML tags and normalizing whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# %%
def load_and_chunk_text(filepath, chunk_size=500, chunk_overlap=50):
    """Load text file, clean it, and split into chunks."""
    # TODO: Read the file
    # TODO: Clean the text (use clean_text function)
    # TODO: Create a RecursiveCharacterTextSplitter
    # TODO: Split and return the chunks
    pass

# %% [markdown]
#
# ## Chunk-Qualität und RAG
#
# - **Gute Chunks** → Gutes Retrieval → Gute Antworten
# - **Schlechte Chunks** → Irrelevante Ergebnisse → Schlechte Antworten
#
# **Beispiel für einen schlechten Chunk**:
# ```
# "Networks sind eine wichtige Methode im Machine Learning.
# Sie sind inspiriert von der Funktionsweise"
# ```
# → Abgeschnitten mitten im Satz!
#
# **Beispiel für einen guten Chunk**:
# ```
# "Neural Networks sind eine wichtige Methode im Machine Learning.
# Sie sind inspiriert von der Funktionsweise des Gehirns."
# ```
# → Vollständiger Gedanke, sinnvolle Einheit

# %% [markdown]
#
# ## Zusammenfassung: Text-Chunking
#
# - **Chunking**: Lange Texte in sinnvolle Stücke aufteilen
# - **RecursiveCharacterTextSplitter**: Intelligente Aufteilung (empfohlen)
# - **chunk_size**: Maximale Chunk-Größe (typisch 500–1000)
# - **chunk_overlap**: Überlappung für Kontext am Rand (typisch 50–200)
# - **Chunk-Qualität** beeinflusst direkt die RAG-Ergebnisse
#
# **Nächster Schritt**: Vector Embeddings — wie finden wir relevante Chunks?
