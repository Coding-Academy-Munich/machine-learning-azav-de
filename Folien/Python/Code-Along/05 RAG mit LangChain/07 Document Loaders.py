# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Document Loaders</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Das Problem: Externe Dokumente
#
# Unser Chatbot kann Fragen beantworten, aber:
#
# - Er kennt nur sein **Trainingswissen**
# - Er kann nicht auf **unsere Dokumente** zugreifen
# - Er weiß nichts über **unser Unternehmen**
#
# **Was wir brauchen**: Daten aus verschiedenen Quellen laden!

# %% [markdown]
#
# ## Typische Quellen
#
# - **PDF-Dateien**: Handbücher, Berichte, Verträge
# - **Textdateien**: Dokumentation, Notizen
# - **Webseiten**: Online-Dokumentation, Artikel
# - **Word-Dokumente**: Geschäftsdokumente
# - **CSV/Excel**: Daten, Tabellen

# %% [markdown]
#
# ## Die Lösung: Document Loaders
#
# LangChain bietet **Document Loaders** für viele Formate:
#
# - `TextLoader`: Textdateien
# - `PyPDFLoader`: PDF-Dateien
# - `WebBaseLoader`: Webseiten
# - `CSVLoader`: CSV-Dateien
# - Und viele mehr...

# %%
import os
from dotenv import load_dotenv

load_dotenv()

# %%
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ml-azav-course"

# %% [markdown]
#
# ## Document: Das Standardformat
#
# Alle Loader geben **Document**-Objekte zurück:
#
# ```python
# Document(
#     page_content="Der Text des Dokuments...",
#     metadata={"source": "datei.txt", "page": 1}
# )
# ```
#
# - `page_content`: Der eigentliche Text
# - `metadata`: Informationen über die Quelle

# %%
from langchain_core.documents import Document

# %% [markdown]
#
# ## TextLoader: Textdateien laden
#
# Der einfachste Loader für `.txt` Dateien:

# %%
from langchain_community.document_loaders import TextLoader

# %%
sample_text = """Python für Einsteiger

Python ist eine beliebte Programmiersprache, die 1991 von Guido van Rossum
entwickelt wurde. Sie ist bekannt für ihre einfache, lesbare Syntax.

Hauptmerkmale:
- Einfach zu lernen
- Vielseitig einsetzbar
- Große Community
- Umfangreiche Bibliotheken

Python wird verwendet für:
1. Webentwicklung
2. Datenanalyse
3. Künstliche Intelligenz
4. Automatisierung
"""

sample_file = "sample_document.txt"
with open(sample_file, "w", encoding="utf-8") as f:
    f.write(sample_text)

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Das Document-Objekt untersuchen

# %%

# %%

# %%

# %% [markdown]
#
# ## WebBaseLoader: Webseiten laden
#
# Für das Laden von Webseiten:

# %%
# !pip install beautifulsoup4

# %%
from langchain_community.document_loaders import WebBaseLoader

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## PyPDFLoader: PDF-Dateien laden
#
# Für PDF-Dokumente:

# %%
# !pip install pypdf

# %%
from langchain_community.document_loaders import PyPDFLoader

# %% [markdown]
#
# PDFs werden **seitenweise** geladen:
#
# ```python
# loader = PyPDFLoader("dokument.pdf")
# pages = loader.load()  # Liste von Documents, eine pro Seite
#
# for page in pages:
#     print(f"Seite {page.metadata['page']}")
#     print(page.page_content[:100])
# ```

# %% [markdown]
#
# ## Mehrere Dokumente laden
#
# Oft haben wir mehrere Dateien:

# %%
sample_texts = {
    "python_basics.txt": "Python Grundlagen: Variablen, Listen, Schleifen...",
    "python_functions.txt": "Funktionen in Python: def, Parameter, return...",
    "python_classes.txt": "Klassen in Python: class, __init__, Methoden...",
}

for filename, content in sample_texts.items():
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

# %%
all_documents = []
for filename in sample_texts.keys():
    loader = TextLoader(filename, encoding="utf-8")
    docs = loader.load()
    all_documents.extend(docs)

# %%

# %%

# %% [markdown]
#
# ## DirectoryLoader: Ganze Ordner laden
#
# Für viele Dateien auf einmal:

# %%
from langchain_community.document_loaders import DirectoryLoader

# %% [markdown]
#
# ```python
# loader = DirectoryLoader(
#     "docs/",
#     glob="**/*.txt",  # Nur .txt Dateien
#     loader_cls=TextLoader,
#     loader_kwargs={"encoding": "utf-8"}
# )
# documents = loader.load()
# ```

# %% [markdown]
#
# ## Token zählen
#
# Wichtig für Kosten und Context-Länge:

# %%
# !pip install tiktoken

# %%
import tiktoken


def count_tokens(text, model="gpt-4"):
    """Count tokens in text for a given model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# %%

# %% [markdown]
#
# ## Das nächste Problem: Zu lange Dokumente
#
# Ein typisches Problem:
#
# - PDF-Handbuch: **50.000+ Tokens**
# - Webseite: **10.000+ Tokens**
# - LLM Context-Limit: **4.096 - 128.000 Tokens**
#
# **Lösung**: Dokumente in **Chunks** aufteilen!
#
# → Das lernen wir im nächsten Kapitel (Text Processing)

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Document Loaders** laden Daten aus verschiedenen Quellen
# - **Document-Objekte**: `page_content` + `metadata`
# - **TextLoader**: Textdateien
# - **PyPDFLoader**: PDFs (seitenweise)
# - **WebBaseLoader**: Webseiten
# - **DirectoryLoader**: Ganze Ordner
# - **Token-Zählung** wichtig für Kosten/Limits

# %% [markdown]
#
# ## Workshop: Document Inspector
#
# **Ziel**: Ein Tool zum Untersuchen von Dokumenten!
#
# **Aufgaben**:
# 1. Textdateien laden und anzeigen
# 2. URLs laden und anzeigen
# 3. Token-Anzahl berechnen
# 4. Gradio Interface mit Datei-Upload und URL-Eingabe

# %%
import gradio as gr

# %% [markdown]
#
# ### Teil 1: Dokument-Analyse Funktionen

# %%
def analyze_document(doc):
    """Analyze a single document."""
    content = doc.page_content
    tokens = count_tokens(content)
    chars = len(content)
    lines = content.count("\n") + 1
    words = len(content.split())

    return {
        "content": content,
        "tokens": tokens,
        "characters": chars,
        "lines": lines,
        "words": words,
        "metadata": doc.metadata,
    }


# %%
def load_and_analyze_text(file_path):
    """Load text file and analyze it."""
    # TODO: Load file with TextLoader and analyze
    pass


# %%
def load_and_analyze_url(url):
    """Load URL and analyze it."""
    # TODO: Load URL with WebBaseLoader and analyze
    pass


# %% [markdown]
#
# ### Teil 2: Testen

# %%

# %%

# %% [markdown]
#
# ### Teil 3: Gradio Interface

# %%
# TODO: Create Gradio interface for document inspector

# %%

# %% [markdown]
#
# ## Workshop-Aufgaben Zusammenfassung
#
# ### Basis (Pflicht):
# 1. TextLoader für Dateien verwenden
# 2. WebBaseLoader für URLs verwenden
# 3. Token-Zählung implementieren
# 4. Gradio Interface erstellen
#
# ### Erweitert (Optional):
# 5. PDF-Support hinzufügen
# 6. Keyword-Suche in geladenen Dokumenten
# 7. Mehrere Dateien gleichzeitig analysieren

# %% [markdown]
#
# ## Ausblick: Text Processing und RAG
#
# Mit Document Loaders können wir Daten laden.
#
# **Nächste Schritte**:
# 1. **Text Processing**: Dokumente in Chunks aufteilen
# 2. **Vector Embeddings**: Text in Vektoren umwandeln
# 3. **Vector Databases**: Vektoren speichern und suchen
# 4. **RAG**: Retrieval Augmented Generation
#
# Das ist die Basis für **Dokumenten-Chatbots**!

# %%
os.remove("sample_document.txt")
for filename in sample_texts.keys():
    os.remove(filename)

# %%
