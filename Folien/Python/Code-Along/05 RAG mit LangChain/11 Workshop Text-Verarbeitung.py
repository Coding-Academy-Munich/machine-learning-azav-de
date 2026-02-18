# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: Text-Verarbeitung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Workshop: Text-Verarbeitung
#
# In diesem Workshop üben Sie die Schritte der Text-Verarbeitung:
# 1. Regex-basierte Bereinigung mit `clean_text()`
# 2. Encoding-Probleme reparieren mit `ftfy`
# 3. Web-Text normalisieren mit `clean-text`
# 4. Haupttext aus HTML extrahieren mit `trafilatura`
# 5. Text bereinigen und aufteilen
# 6. Mit Chunk-Parametern experimentieren
# 7. Verschiedene Splitter vergleichen

# %%
import re
import ftfy
import trafilatura
from cleantext import clean
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


# %%
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# %% [markdown]
#
# ### Aufgabe 1: Datei bereinigen mit `clean_text()`
#
# - Lesen Sie die Datei `docs/messy_report.txt`
# - Wenden Sie `clean_text()` auf den Inhalt an
# - Geben Sie den Text vorher und nachher aus

# %%
with open("docs/messy_report.txt", "r") as f:
    messy_report = f.read()

print("=== Before ===")
print(messy_report)

# TODO: Apply clean_text() and print the result

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 2: Encoding-Probleme reparieren mit ftfy
#
# - Lesen Sie die Datei `docs/encoding_problems.txt`
# - Verwenden Sie `ftfy.fix_text()` um die HTML-Entities zu reparieren
# - Kombinieren Sie `ftfy.fix_text()` mit `clean_text()`

# %%
with open("docs/encoding_problems.txt", "r") as f:
    encoded_text = f.read()

print("=== Before ===")
print(encoded_text)

# TODO: Apply ftfy.fix_text() and clean_text(), print the result

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 3: Web-Text normalisieren mit clean-text
#
# - Lesen Sie die Datei `docs/scraped_content.txt`
# - Verwenden Sie `clean()` mit `no_urls=True`, `no_emails=True`, `no_emoji=True`
# - Probieren Sie auch Platzhalter: `replace_with_url="REPLACED-URL"`,
#   `replace_with_email="REPLACED-EMAIL"`

# %%
with open("docs/scraped_content.txt", "r") as f:
    scraped = f.read()

print("=== Before ===")
print(scraped)

# TODO: Apply clean() with different options, print the results

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 4: Haupttext extrahieren mit trafilatura
#
# - Lesen Sie die Datei `docs/sample_blog_post.html`
# - Verwenden Sie `trafilatura.extract()` um den Haupttext zu extrahieren
# - Geben Sie das Ergebnis aus

# %%
with open("docs/sample_blog_post.html", "r") as f:
    html_content = f.read()

print("=== Raw HTML (first 500 chars) ===")
print(html_content[:500])

# TODO: Use trafilatura.extract() to extract the main text and print it

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 5: Text bereinigen und aufteilen
#
# Schreiben Sie eine Funktion `clean_and_chunk(filepath)`, die:
# 1. Eine Textdatei liest
# 2. Den Text mit `clean_text()` bereinigt
# 3. Den bereinigten Text in Chunks aufteilt (`chunk_size=500`, `chunk_overlap=50`)
# 4. Die Chunks zurückgibt
#
# Testen Sie Ihre Funktion mit `docs/ai-intro.txt`.

# %%
def clean_and_chunk(filepath, chunk_size=500, chunk_overlap=50):
    # TODO: Read the file
    # TODO: Clean the text with clean_text()
    # TODO: Create a RecursiveCharacterTextSplitter
    # TODO: Split and return the chunks
    pass


# %%

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 6: Chunk-Parameter experimentieren
#
# Verwenden Sie den folgenden längeren Text.
# Erstellen Sie einen `RecursiveCharacterTextSplitter` und testen Sie verschiedene
# Kombinationen von `chunk_size` und `chunk_overlap`:
#
# - chunk_size: 100, 200, 500
# - chunk_overlap: 0, 20%, 50%
#
# Geben Sie für jede Kombination die Anzahl der Chunks und die ersten beiden Chunks aus.
# Beobachten Sie: Wie verändern sich die Chunks?

# %%
long_text = """\
Machine Learning ist ein Teilgebiet der künstlichen Intelligenz, das sich mit \
dem automatischen Lernen aus Daten beschäftigt. Anstatt explizite Regeln zu \
programmieren, lernen ML-Algorithmen Muster aus Beispieldaten.

Supervised Learning ist die häufigste Form des Machine Learning. Das Modell \
lernt aus gelabelten Trainingsdaten eine Zuordnung von Eingabe zu Ausgabe. \
Beispiele sind Klassifikation und Regression.

Bei der Klassifikation ordnet das Modell Eingaben einer von mehreren Kategorien \
zu. Zum Beispiel: Ist diese E-Mail Spam oder kein Spam? Ist dieses Bild eine \
Katze oder ein Hund?

Bei der Regression sagt das Modell einen numerischen Wert vorher. Zum Beispiel: \
Wie hoch ist der Preis dieses Hauses? Wie viele Produkte werden nächsten Monat \
verkauft?

Neural Networks sind eine besonders leistungsfähige Methode des Machine Learning. \
Sie bestehen aus Schichten von künstlichen Neuronen, die komplexe Muster erkennen \
können. Deep Learning nutzt besonders tiefe Neural Networks mit vielen Schichten.

Large Language Models wie GPT und Claude sind sehr große Neural Networks, die auf \
Milliarden von Texten trainiert wurden. Sie können Texte verstehen, generieren \
und Fragen beantworten.\
"""

# %%
# Experimentieren Sie mit verschiedenen Parametern:
# Experiment with different parameters:
chunk_sizes = [100, 200, 500]
overlap_percentages = [0, 20, 50]

# TODO: Erstellen Sie für jede Kombination einen Splitter
# TODO: Geben Sie Anzahl Chunks und ersten Chunk aus
# TODO: Create a splitter for each combination
# TODO: Print number of chunks and first chunk

# %% [markdown]
#
# ### Aufgabe 7: Text-Splitter vergleichen
#
# Vergleichen Sie `CharacterTextSplitter` und `RecursiveCharacterTextSplitter`
# auf demselben Text:
#
# 1. Erstellen Sie beide Splitter mit `chunk_size=200` und `chunk_overlap=20`
# 2. Splitten Sie den Text mit beiden
# 3. Vergleichen Sie die Ergebnisse: Welcher Splitter erzeugt bessere Chunks?
#
# **Tipp**: `CharacterTextSplitter` braucht den Parameter `separator="\n"`.

# %%
# TODO: Erstellen Sie beide Splitter
# TODO: Splitten Sie den Text
# TODO: Vergleichen Sie die Ergebnisse
# TODO: Create both splitters
# TODO: Split the text
# TODO: Compare the results

# %%

# %%

# %%
