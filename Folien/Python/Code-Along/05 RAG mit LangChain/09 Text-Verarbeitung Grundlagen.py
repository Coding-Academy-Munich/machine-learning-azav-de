# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Text-Verarbeitung: Grundlagen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Warum Text-Verarbeitung?
#
# - Wir haben RAG kennengelernt und wissen, wie man Dokumente lädt
# - **Problem**: Dokumente sind oft zu lang für LLM-Context-Windows
# - **Lösung**: Dokumente in kleinere **Chunks** (Stücke) aufteilen
# - Dann nur die **relevanten** Chunks für jede Frage finden

# %%

# %%
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

# %% [markdown]
#
# ## Text laden und aufbereiten
#
# - Text aus Dateien lesen
# - Unnötige Zeichen entfernen
# - In handhabbare Stücke teilen
# - LangChain macht vieles davon automatisch!

# %% [markdown]
#
# Beispiel-Text (simuliert ein Dokument)

# %%
document = """
Machine Learning ist ein Teilgebiet der künstlichen Intelligenz.
Es beschäftigt sich mit dem automatischen Lernen aus Daten.

Neural Networks sind eine wichtige Methode im Machine Learning.
Sie sind inspiriert von der Funktionsweise des Gehirns.

Large Language Models verwenden sehr große Neural Networks.
Sie werden auf riesigen Textmengen trainiert.
"""

# %%

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
# ## LangChain Text Splitters
#
# - **CharacterTextSplitter**: Teilt bei bestimmten Zeichen
# - **RecursiveCharacterTextSplitter**: Schlauer (Absätze → Sätze → Wörter)
# - **Empfohlen**: RecursiveCharacterTextSplitter

# %%
# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # Maximum chunk size
    chunk_overlap=50,      # Overlap between chunks
    length_function=len,   # How to measure length
    separators=["\n\n", "\n", ". ", " ", ""]  # Split at these in order
)

# %%

# %% [markdown]
#
# ## Chunk-Parameter verstehen
#
# - **chunk_size**: Maximale Größe eines Chunks (in Zeichen)
# - **chunk_overlap**: Überlappung zwischen Chunks
#   - Warum? Kontext am Rand nicht verlieren!
# - **separators**: Wo wird geteilt? (Absatz → Satz → Wort)
#
# **Typische Werte**:
# - chunk_size: 500-1000 für Dokumente
# - chunk_overlap: 50-200 (10-20% von chunk_size)

# %% [markdown]
#
# ## Tokens vs. Zeichen
#
# - Erinnerung: LLMs zählen in **Tokens**, nicht Zeichen
# - **Faustregel**: 1 Token ≈ 4 Zeichen (für englischen Text)
# - LangChain bietet auch einen **Token-basierten Splitter**

# %%
from langchain_text_splitters import TokenTextSplitter

# Create token-based splitter
token_splitter = TokenTextSplitter(
    chunk_size=50,  # in tokens
    chunk_overlap=10
)


# %%

# %% [markdown]
#
# ## Praktische Text-Operationen
#
# - Text aus Dateien lesen
# - Bereinigen (lowercase, Sonderzeichen)
# - In Chunks aufteilen mit LangChain
# - Für RAG vorbereiten

# %%
def load_and_chunk_text(filepath, chunk_size=500, chunk_overlap=50):
    """Load text file and split into chunks"""
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split
    chunks = splitter.split_text(text)

    return chunks

# %% [markdown]
#
# ## Workshop-Aufgaben
#
# 1. Text aus Datei laden (mit Document Loader oder manuell)
# 2. Mit verschiedenen chunk_size experimentieren
# 3. Chunk-Overlap ausprobieren
# 4. Verschiedene Splitter vergleichen

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Chunking**: Lange Texte in Stücke teilen
# - **RecursiveCharacterTextSplitter**: Intelligente Aufteilung
# - **chunk_size**: Maximale Chunk-Größe
# - **chunk_overlap**: Überlappung für Kontext am Rand
# - **TokenTextSplitter**: Alternative für token-basierte Aufteilung
#
# **Nächster Schritt**: Vector Embeddings - wie finden wir relevante Chunks?
