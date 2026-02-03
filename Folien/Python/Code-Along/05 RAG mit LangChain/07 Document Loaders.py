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

# %%
load_dotenv()

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

# %%

# %% [markdown]
#
# ## TextLoader: Textdateien laden
#
# Der einfachste Loader für `.txt` Dateien:

# %%
from langchain_community.document_loaders import TextLoader

# %%
loader = TextLoader("docs/short.txt", encoding="utf-8")

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
import re

# %%
url = "https://de.wikipedia.org/wiki/Python_(Programmiersprache)"

# %%

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


# %%

# %%

# %%
for page in pdf_pages:
    print(f"Page {page.metadata['page']}, length: {len(page.page_content)} characters")
    print(page.page_content[:50].replace("\n", " "))

# %% [markdown]
#
# Beachten Sie den Unterschied, wenn wir den gleichen Text aus einer Textdatei
# laden:

# %%

# %%

# %%
for page in text_pages:
    print(f"Page: {page.metadata.get('page')}, length: {len(page.page_content)} characters")
    print(page.page_content[:50].replace("\n", " "))

# %% [markdown]
#
# ## Mehrere Dokumente laden
#
# Oft haben wir mehrere Dateien:

# %%
sample_texts = ["docs/python-intro.txt", "docs/python-functions.txt", "docs/python-classes.txt"]

# %%
all_documents = []

# %%

# %%

# %%
for doc in all_documents:
    print(f"- {doc.metadata['source']}:\t{doc.page_content[:40]}...")

# %% [markdown]
#
# ## DirectoryLoader: Ganze Ordner laden
#
# Für viele Dateien auf einmal:

# %%
from langchain_community.document_loaders import DirectoryLoader

# %%

# %%

# %%
for doc in all_documents:
    print(f"- {doc.metadata['source']}:\t{doc.page_content[:40]}...")


# %% [markdown]
#
# ## Token zählen
#
# - Interessant für Kosten und Context-Länge.
# - Wir verwenden die `tiktoken` Bibliothek.
#   - Verschiedene Modelle verwenden unterschiedliche Tokenisierung.
#   - Tiktoken unterstützt mehrere Modelle, aber nicht `ministral`.

# %%
# !pip install tiktoken

# %%
import tiktoken


# %%

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
# - **Token-Zählung** um Kosten/Limits zu überwachen

# %% [markdown]
#
# ## Workshop-Vorbereitung
#
# ### Gradio Blocks, Tabs und Buttons
#
# Bevor wir den Workshop starten, lernen wir die Gradio-Konzepte für das
# Interface:
#
# - **Blocks**: Container für komplexe Layouts
# - **Tabs**: Mehrere Seiten in einem Interface
# - **Buttons**: Lösen Aktionen mit Callbacks aus

# %% [markdown]
#
# ### Einfaches Beispiel: Text-Transformator
#
# Ein Interface mit zwei Tabs für verschiedene Text-Transformationen:

# %% [markdown]
#
# ### Neue Gradio Konzepte
#
# - `gr.Blocks()`: Erstellt einen Container für das Layout
# - `with gr.Tab("Name")`: Erstellt einen Tab mit dem angegebenen Namen
# - `gr.Button("Text")`: Erstellt einen Button
# - `button.click(fn, inputs, outputs)`: Verbindet Button mit einer Funktion
#   - `fn`: Die Funktion, die aufgerufen wird
#   - `inputs`: Die Eingabe-Komponenten
#   - `outputs`: Die Ausgabe-Komponenten

# %%
def to_uppercase(text):
    """Convert text to uppercase."""
    return text.upper()


# %%
def to_leetspeak(text):
    """Convert text to leetspeak."""
    replacements = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
    result = text.lower()
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    return result


# %%
import gradio as gr

# %%
with gr.Blocks(title="Text Transformer") as text_demo:
    gr.Markdown("# Text Transformer")

    with gr.Tab("Uppercase"):
        upper_input = gr.Textbox(label="Enter text", placeholder="Hello World...")
        upper_button = gr.Button("Convert to UPPERCASE")
        upper_output = gr.Textbox(label="Result")
        upper_button.click(to_uppercase, inputs=upper_input, outputs=upper_output)

    with gr.Tab("Leetspeak"):
        leet_input = gr.Textbox(label="Enter text", placeholder="Hello World...")
        leet_button = gr.Button("Convert to L33tsp34k")
        leet_output = gr.Textbox(label="Result")
        leet_button.click(to_leetspeak, inputs=leet_input, outputs=leet_output)

# %%
# text_demo.launch()

# %%

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
# ### Teil 1: Dokument-Analyse Funktion
#
# Implementieren Sie eine Funktion `analyze_document(doc)`, die ein `Document`-Objekt
# analysiert und Statistiken zurückgibt.
#
# **Eingabe**: Ein `Document`-Objekt mit:
# - `page_content`: Der Textinhalt des Dokuments
# - `metadata`: Ein Dictionary mit Metadaten (z.B. `source`)
#
# **Aufgaben**:
# 1. Zählen Sie die **Tokens** mit der `count_tokens()`-Funktion
# 2. Zählen Sie die **Zeichen** (Länge des Textes)
# 3. Zählen Sie die **Zeilen** (Anzahl der Zeilenumbrüche + 1)
# 4. Zählen Sie die **Wörter** (Text mit `split()` aufteilen)
#
# **Rückgabe**: Ein Dictionary mit den Schlüsseln:
# - `"content"`: Der Textinhalt
# - `"tokens"`: Anzahl der Tokens (GPT-5)
# - `"characters"`: Anzahl der Zeichen
# - `"lines"`: Anzahl der Zeilen
# - `"words"`: Anzahl der Wörter
# - `"metadata"`: Die Metadaten des Dokuments

# %%
def analyze_document(doc):
    """Analyze a single document."""
    pass


# %% [markdown]
#
# ### Teil 2: Textdatei laden und analysieren
#
# Implementieren Sie eine Funktion `load_and_analyze_text(file_path)`, die eine
# Textdatei lädt und analysiert.
#
# **Schritte**:
# 1. Erstellen Sie einen `TextLoader` mit dem Dateipfad und `encoding="utf-8"`
# 2. Laden Sie die Dokumente mit `.load()`
# 3. Falls Dokumente vorhanden sind, rufen Sie `analyze_document()` auf dem ersten auf
# 4. Geben Sie das Ergebnis zurück
#
# **Fehlerbehandlung**:
# - Bei leerer Dokumentenliste: `{"error": "Keine Dokumente geladen"}` zurückgeben
# - Bei Ausnahmen: `{"error": str(e)}` zurückgeben (mit try/except)

# %%
def load_and_analyze_text(file_path):
    """Load text file and analyze it."""
    pass


# %% [markdown]
#
# ### Teil 3: URL laden und analysieren
#
# Implementieren Sie eine Funktion `load_and_analyze_url(url)`, die eine Webseite
# lädt und analysiert.
#
# **Schritte**:
# 1. Erstellen Sie einen `WebBaseLoader` mit der URL
# 2. Laden Sie die Dokumente mit `.load()`
# 3. Falls Dokumente vorhanden sind, rufen Sie `analyze_document()` auf dem ersten auf
# 4. Geben Sie das Ergebnis zurück
#
# **Fehlerbehandlung** (wie bei `load_and_analyze_text`):
# - Bei leerer Dokumentenliste: `{"error": "Keine Dokumente geladen"}` zurückgeben
# - Bei Ausnahmen: `{"error": str(e)}` zurückgeben

# %%
def load_and_analyze_url(url):
    """Load URL and analyze it."""
    pass


# %% [markdown]
#
# ### Teil 4: Testen
#
# Testen Sie die Funktionen `load_and_analyze_text()` und `load_and_analyze_url()`:
#
# - Analysieren Sie die Datei `docs/python-intro.txt`
# - Analysieren Sie die URL `https://de.wikipedia.org/wiki/Python_(Programmiersprache)`

# %%

# %%

# %%

# %% [markdown]
#
# ### Teil 5: Gradio Interface
#
# Erstellen Sie ein Gradio-Interface für den Document Inspector mit zwei Tabs.
#
# **Funktion `inspect_file(file)`**:
# - Prüfen Sie, ob `file` `None` ist → Meldung "Bitte laden Sie eine Datei hoch."
# - Rufen Sie `load_and_analyze_text(file.name)` auf
# - Prüfen Sie, ob `"error"` im Ergebnis ist → Fehlermeldung anzeigen
# - Formatieren Sie die Ausgabe mit Statistiken und Vorschau (erste 1000 Zeichen)
#
# **Funktion `inspect_url(url)`**:
# - Prüfen Sie, ob die URL leer ist → Meldung "Bitte geben Sie eine URL ein."
# - Rufen Sie `load_and_analyze_url(url.strip())` auf
# - Prüfen Sie, ob `"error"` im Ergebnis ist → Fehlermeldung anzeigen
# - Formatieren Sie die Ausgabe mit Statistiken und Vorschau
#
# **Gradio Blocks Interface**:
# - Titel: "Document Inspector"
# - Tab 1: Datei-Upload mit `gr.File` (nur `.txt`) und "Analysieren"-Button
# - Tab 2: URL-Eingabe mit `gr.Textbox` und "Laden und Analysieren"-Button
# - Ausgabe jeweils als `gr.Markdown`

# %%
def inspect_file(file):
    """Inspect uploaded file."""
    pass


def inspect_url(url):
    """Inspect URL content."""
    pass


# TODO: Create Gradio Blocks interface with two tabs

# %%

# %%
