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
# 1. Text bereinigen und aufteilen
# 2. Mit Chunk-Parametern experimentieren
# 3. Verschiedene Splitter vergleichen

# %%
import re
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# %% [markdown]
#
# ## Aufgabe 1: Text bereinigen und aufteilen
#
# Schreiben Sie eine Funktion `clean_and_chunk(filepath)`, die:
# 1. Eine Textdatei liest
# 2. Den Text mit `clean_text()` bereinigt
# 3. Den bereinigten Text in Chunks aufteilt (`chunk_size=500`, `chunk_overlap=50`)
# 4. Die Chunks zurückgibt
#
# Testen Sie Ihre Funktion mit `docs/ai-intro.txt`.

# %%
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
# ## Aufgabe 2: Chunk-Parameter experimentieren
#
# Verwenden Sie den folgenden längeren Text.
# Erstellen Sie einen `RecursiveCharacterTextSplitter` und testen Sie verschiedene
# Kombinationen von `chunk_size` und `chunk_overlap`:
#
# - chunk_size: 100, 200, 500
# - chunk_overlap: 0, 20, 50
#
# Geben Sie für jede Kombination die Anzahl der Chunks und den ersten Chunk aus.
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
overlaps = [0, 20, 50]

# TODO: Erstellen Sie für jede Kombination einen Splitter
# TODO: Geben Sie Anzahl Chunks und ersten Chunk aus
# TODO: Create a splitter for each combination
# TODO: Print number of chunks and first chunk

# %% [markdown]
#
# ## Aufgabe 3: Text-Splitter vergleichen
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
