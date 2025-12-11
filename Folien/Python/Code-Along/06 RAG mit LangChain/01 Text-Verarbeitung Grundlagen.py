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
# - **RAG-Systeme** brauchen Texte in Dokumenten
# - Dokumente müssen **vorbereitet** werden
# - In **Chunks** (Stücke) aufteilen
# - Für Vektor-Datenbanken **optimieren**

# %%
import tiktoken

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
# ## Tokenisierung für LLMs
#
# - LLMs verwenden **Subword-Tokenisierung**
# - Nicht Wörter, sondern **Teile von Wörtern**
# - Beispiel: "unhappiness" → ["un", "happiness"]
# - **Wichtig**: Tokens zählen (Kosten!)

# %%
# Count tokens with tiktoken (OpenAI's tokenizer)
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

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
# - **Problem**: chunk_size ist in Zeichen, aber APIs zählen Tokens
# - **Lösung**: Token-basierter Splitter
# - Oder: Mit Zeichen schätzen (1 Token ≈ 4 Zeichen)

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
# 1. Text aus Datei laden
# 2. Mit verschiedenen chunk_size experimentieren
# 3. Chunk-Overlap ausprobieren
# 4. Tokens zählen für Kosten-Schätzung
# 5. Für RAG-System vorbereiten

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Text → Zahlen**: Für ML-Verarbeitung nötig
# - **Tokenisierung**: Text in Tokens aufteilen
# - **Chunking**: Lange Texte in Stücke teilen
# - **LangChain Text Splitters**: Intelligente Aufteilung
# - **chunk_size und chunk_overlap**: Wichtige Parameter
#
# **Nächster Schritt**: RAG-Systeme - Dokumente mit LLMs verbinden!
