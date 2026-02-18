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

import trafilatura
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from visualize_chunks import visualize_chunks

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
print(document[:200] + "\n...\n" + document[-200:])

# %%
print(f"Document length: {len(document)} characters")

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
# with chunk_size=300, chunk_overlap=100

# %%

# %%

# %%
for i, chunk in enumerate(chunks[:3], 1):
    print(f"\n--- Chunk {i} (Länge/length: {len(chunk)}) ---")
    print(chunk)

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
# - chunk_overlap: 50–400 (10–40% von chunk_size)

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
splitter_with_overlap = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=100
)

# %%

# %%

# %%
visualize_chunks(chunks_no_overlap, max_chunks=5)

# %%
visualize_chunks(chunks_with_overlap, max_chunks=5)

# %% [markdown]
#
# ## Problem: Unvollständige Sätze
#
# - Bei unserer jetzigen Chunking-Strategie enden die Chunks oft mitten im Satz
# - Das kann zu unverständlichen oder unvollständigen Informationen führen

# %% [markdown]
#
# ## Verbesserung: Chunking nach Sätzen
#
# - Wir können den `separators`-Parameter anpassen, um zuerst nach Sätzen zu
#   splitten
# - Dadurch erhalten wir Chunks, die eher vollständige Gedanken enthalten
# - Allerdings müssen wir dann größere Chunks bilden, da Sätze oft länger als
#   100 Zeichen sind

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Problem: Manche Splitpunkte werden nicht erkannt
#
# - Unsere Regex für Satzende (`(?<=[.!?:])\s+`) erkennt nicht alle möglichen
#   Satzenden.
# - Zum Beispiel werden Anführungszeichen am Ende eines Satzes nicht
#   berücksichtigt.
# - Daher werden für Chunk 2 und Chunk 3 keine Überlappungen angezeigt, obwohl
#   sie existieren.

# %% [markdown]
#
# ## Lösung: Regex anpassen
#
# - Wir können unsere Regex anpassen, um auch Anführungszeichen am Satzende zu
#   berücksichtigen
# - Dadurch werden Überlappungen korrekt erkannt


# %%
sentence_rx = r"(?:(?<=[.!?:])|(?<=[.!?:][\"']))\s+"

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Tokens vs. Zeichen
#
# - Erinnerung: LLMs zählen in **Tokens**, nicht Zeichen
# - Ein **Token** ist ein Wort-Teilstück, das das Modell als Einheit verarbeitet
# - Beispiel: `"understanding"` → `["under", "standing"]` (2 Tokens)
# - **Faustregel**: 1 Token ≈ 4 Zeichen (für englischen Text)
# - Für deutschen Text etwas weniger (wegen längerer Wörter)

# %% [markdown]
#
# - Beispiel: `"This sentence has 40 characters"` → ca. 7 Tokens
# - **Warum wichtig?** Context-Limits von LLMs sind in Tokens angegeben
# - LangChain bietet auch einen **Token-basierten Splitter**

# %%
token_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=10)

# %%

# %%

# %%

# %% [markdown]
#
# ## Bereinigen + Chunken: Echte Webseiten
#
# - Bisher haben wir mit einer sauberen Textdatei gearbeitet
# - Echte Daten kommen oft aus dem Web und sind **nicht sauber**
# - In der Praxis: Erst bereinigen, dann chunken
# - Wir nutzen die Tools aus dem letzten Abschnitt:
#   - `AsyncHtmlLoader` zum Laden von Webseiten (liefert rohes HTML)
#   - `trafilatura` für eine saubere Text-Extraktion

# %% [markdown]
#
# ## Schritt 1: Webseite laden mit AsyncHtmlLoader
#
# **Hinweis:** Wikipedia blockiert `AsyncHtmlLoader`, daher verwenden wir eine
# alternative Seite mit ähnlichem Inhalt.

# %%
url = "https://grokipedia.com/page/Python_programming_language"

# %%
loader = AsyncHtmlLoader([url])

# %%

# %%
print(docs[0].page_content[:500])

# %% [markdown]
#
# ## Schritt 2: Text extrahieren mit trafilatura
#
# - `AsyncHtmlLoader` liefert **rohes HTML** — nicht geeignet zum Chunken
# - `trafilatura` extrahiert den **Haupttext** und entfernt Boilerplate
#   (Navigation, Fußnoten, etc.)

# %%

# %%
print(text[:500])

# %%
docs[0].page_content = text

# %% [markdown]
#
# ## Problem: Referenzen in Wikipedia-Texten
#
# - Wikipedia/Grokipedia-Texte enthalten Referenzen wie `.[10][11]`
# - Diese stören beim Chunking (Satzende wird nicht erkannt)
# - **Lösung**: Referenzen vor dem Chunking entfernen

# %%
docs[0].page_content = re.sub(r"\[\d+\]", "", docs[0].page_content)

# %%
print(docs[0].page_content[:500])

# %% [markdown]
#
# ## Schritt 3: Bereinigten Text chunken

# %%

# %%

# %%

# %%


# %% [markdown]
#
# ## Chunk-Qualität und RAG
#
# - **Gute Chunks** → Gutes Retrieval → Gute Antworten
# - **Schlechte Chunks** → Irrelevante Ergebnisse → Schlechte Antworten

# %% [markdown]
#
# **Beispiel für einen schlechten Chunk**:
# ```
# "Networks sind eine wichtige Methode im Machine Learning.
# Sie sind inspiriert von der Funktionsweise"
# ```
# → Abgeschnitten mitten im Satz!

# %% [markdown]
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

# %% [markdown]
#
# ## Mini-Workshop: Komplette Pipeline als Funktion
#
# Schreiben Sie eine Funktion `load_and_chunk_url(url, ...)`, die:
# 1. Die Seite mit `AsyncHtmlLoader` lädt
# 2. Den Text mit `trafilatura.extract()` extrahiert
# 3. Referenzen (`[1]`, `[11]`, ...) mit `re.sub()` entfernt
# 4. Den Text mit `RecursiveCharacterTextSplitter` in Chunks aufteilt
# 5. Die Chunks zurückgibt

# %%
def load_and_chunk_url(url, chunk_size=600, chunk_overlap=300):
    """Fetch a web page, extract clean text, and split into chunks."""
    # TODO: Load the page with AsyncHtmlLoader
    # TODO: Extract text with trafilatura.extract()
    # TODO: Remove citations with re.sub()
    # TODO: Create a RecursiveCharacterTextSplitter with sentence_rx
    # TODO: Split and return the chunks
    pass

# %%

# %%

# %%
