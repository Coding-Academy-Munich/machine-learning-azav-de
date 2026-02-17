# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Text bereinigen</b>
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
#
# Aber vorher müssen wir den Text **bereinigen**!

# %%
import re
import dotenv

# %%
dotenv.load_dotenv()  # Load environment variables from .env file

# %% [markdown]
#
# ## Text bereinigen: Warum?
#
# - Texte aus Dateien oder Webseiten haben oft Probleme:
#   - Überflüssige Leerzeichen und Leerzeilen
#   - HTML-Tags (`<br>`, `<p>`, etc.)
#   - Tabulatoren und Sonderzeichen
# - Diese Artefakte stören das Chunking und die Suche
# - **Erster Schritt**: Text säubern, bevor wir ihn aufteilen

# %% [markdown]
#
# ## Beispiel: "Verschmutzter" Text

# %%
dirty_text = """
   Machine Learning ist ein Teilgebiet der KI.
\t\tEs lernt automatisch aus Daten.<br>

      Neurale Netzwerke   sind     eine <em>wichtige</em> Methode.
<p>Sie sind
   vom Gehirn
   `inspiriert'.</p>



   „Large Language Models‟    verwenden sehr große Netzwerke.
"""


# %%

# %% [markdown]
#
# ## Text bereinigen: Schritt für Schritt
#
# 1. **HTML-Tags entfernen**: `re.sub(r'<[^>]+>', '', text)`
# 2. **Whitespace normalisieren** (Tabs, mehrfache Leerzeichen): `re.sub(r'[^\S\n]+', ' ', text)`
# 3. **Leerzeichen am Zeilenanfang/-ende entfernen**
# 4. **Mehrfache Leerzeilen reduzieren**: `re.sub(r'\n{3,}', '\n\n', text)`
# 5. **Leerzeichen am Rand entfernen**: `text.strip()`

# %%
def clean_text(text):
    """Clean text by removing HTML tags and normalizing whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    # text = text.replace("\t", " ")
    # text = re.sub(r" +", " ", text)
    # Better: Normalize all kinds of whitespace (tabs, multiple spaces) except
    # newlines in one step:
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n +", "\n", text)  # Remove spaces at beginning of lines
    text = re.sub(r" +\n", "\n", text)  # Remove spaces at end of lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# %%

# %%

# %% [markdown]
#
# **Hinweis:** Text cleaning ist stark vom Format der Rohtexte abhängig.
#
# Wir können `clean_text` erweitern mit zusätzlichen Schritten:
# - Ersetzen von einfachen Zeilenumbrüchen (`\n`) durch Leerzeichen
# - Ersetzen von verschiedenen Arten von Anführungszeichen durch einfache
#   Anführungszeichen (`'`)


# %%
def clean_text_2(text):
    text = clean_text(text)
    text = re.sub(r"([^\n])\n([^\n])", r"\1 \2", text)
    text = re.sub(r'[""«»„‟`"]', "'", text)
    text = re.sub(r" +", " ", text)
    return text

# %%

# %% [markdown]
#
# ## Bereinigen von Dokumenten und LangChain Page Loader
#
# - Viele Dokumenten-Loader (z.B. WebBaseLoader) verwenden Beautiful Soup, um
#   HTML-Tags zu entfernen
# - Das Ergebnis ist oft schon bereinigt, aber manchmal bleiben Artefakte übrig


# %%
from langchain_community.document_loaders import WebBaseLoader

# %%
loader = WebBaseLoader("https://de.wikipedia.org/wiki/Python_(Programmiersprache)")

# %%

# %%

# %% [markdown]
#
# Dokument nach dem Laden (bereinigt mit Beautiful Soup):

# %%

# %% [markdown]
#
# Dokument bereinigt mit unserer `clean_text`-Funktion:

# %%

# %% [markdown]
#
# Wir können den `page_content` von Dokumenten modifizieren, wenn wir
# zusätzliche Nachbearbeitung vornehmen wollen:

# %%

# %%


# %% [markdown]
#
# ## Besser: Verwenden von Bibliotheken
#
# Es gibt viele Bibliotheken, die Textbereinigung und -normalisierung unterstützen:
#
# - **BeautifulSoup**: HTML-Tags entfernen, Text extrahieren
# - **inscriptis**: HTML in lesbaren Text umwandeln (z.B. `&nbsp;` → Leerzeichen)
# - **ftfy**: "Fixes Text For You" — repariert kaputte Unicode-Zeichen, falsche Anführungszeichen, etc.
# - **clean-text**: Entfernt HTML, URLs, Emojis, normalisiert Unicode, etc.
# - **trafilatura**: Extrahiert den Haupttext aus Webseiten, entfernt Boilerplate, etc.
# - **newspaper4k**: Ähnlich wie trafilatura, aber mit Fokus auf Nachrichtenartikel

# %%
# !pip install clean-text ftfy trafilatura unidecode --root-user-action=ignore

# %%
import requests
import trafilatura
import ftfy
from cleantext import clean

# %%
headers = {"User-Agent": "PythonCourse/1.0 (Educational purposes)"}

# %%
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
from textwrap import wrap


# %%
def wrap_text(text, width=72):
    lines = text.splitlines()
    wrapped_lines = [wrap(line, width=width) for line in lines]
    return "\n\n".join(["\n".join(line) for line in wrapped_lines])

# %%

# %% [markdown]
#
# ## Text-Encodings
#
# - Text wird in Computern als **Bytes** gespeichert
# - Ein Byte kann nur 256 verschiedene Werte speichern, aber es gibt viel mehr
#   Zeichen (z.B. Umlaute, Emojis)
# - We need a way to map (multiple) bytes to characters
#   - **UTF-8**: Moderner Standard, unterstützt alle Zeichen (Umlaute, Emojis,
#     ...)
#   - **Latin-1** (ISO 8859-1): Älterer Standard, nur westeuropäische Zeichen


# %% [markdown]
#
# ## Wie entstehen Encoding-Fehler?
#
# - **Problem**: Text wird als UTF-8 **gespeichert**, aber als Latin-1 **gelesen**
#   - Oder umgekehrt!
# - Ergebnis: **Mojibake** -- unlesbarer Zeichensalat

# %% [markdown]
#
# ## Encoding-Fehler simulieren

# %%
original = "Schöne Grüße aus Österreich!"

# %%
as_bytes = original.encode("utf-8")

# %%

# %% [markdown]
#
# Wenn diese UTF-8 Bytes als Latin-1 interpretiert werden:

# %%
broken = as_bytes.decode("latin-1")

# %%

# %% [markdown]
#
# ## ftfy: "Fixes Text For You"
#
# - **ftfy** erkennt und repariert solche Encoding-Fehler automatisch
# - Funktioniert auch bei doppelter Fehlkodierung
# - Repariert auch HTML-Entities und Windows-1252 Probleme

# %%

# %% [markdown]
#
# HTML-Entities aus Web-Scraping:

# %%
html_text = "Preis: 50&#x20AC; f&#252;r das B&#252;ro"

# %%

# %%

# %% [markdown]
#
# URL-kodierte Zeichen (brauchen `urllib.parse`, nicht ftfy):

# %%
from urllib.parse import unquote

# %%
url_text = "Suche: K%C3%BCnstliche%20Intelligenz"

# %%

# %%

# %% [markdown]
#
# ## Text normalisieren mit clean-text
#
# - **clean-text** kombiniert viele Bereinigungsschritte in einer Funktion:
#   - Unicode reparieren (wie ftfy)
#   - URLs, E-Mails, Emojis entfernen oder ersetzen
#   - Whitespace normalisieren
#   - Optionale Konvertierung zu ASCII und Kleinbuchstaben
# - Besonders nützlich für Text aus Web-Scraping

# %%
scraped_text = """\
Die Firma\t\t(gegründet 2019)   bietet KI-Lösungen an.
Kontakt: info@example.com  oder https://www.example.com/kontakt
Mehr Infos auf  https://blog.example.com/ai-trends-2024  😊👍
Copyright © 2024   Firma GmbH • Alle Rechte vorbehalten"""

# %%

# %%
cleaned = clean(
    scraped_text,
    no_urls=True,
    no_emails=True,
    no_emoji=True,
    to_ascii=False,
    lower=False,
)

# %%

# %% [markdown]
#
# Mit Platzhaltern statt Entfernung:

# %%
cleaned_with_placeholders = clean(
    scraped_text,
    no_urls=True,
    no_emails=True,
    no_emoji=True,
    to_ascii=False,
    lower=False,
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
)


# %%

# %% [markdown]
#
# ## Bereinigungspipeline für LangChain-Dokumente
#
# - Wir haben verschiedene Bereinigungstechniken kennengelernt
# - Jetzt kombinieren wir sie zu einer **wiederverwendbaren Funktion**
# - Funktioniert mit **jedem** Document Loader (Web, PDF, etc.)

# %%
def clean_documents(docs, clean_fn=clean_text_2):
    for doc in docs:
        doc.page_content = clean_fn(doc.page_content)
    return docs


# %% [markdown]
#
# Verwendung mit WebBaseLoader:

# %%
loader = WebBaseLoader("https://de.wikipedia.org/wiki/Python_(Programmiersprache)")


# %%

# %%

# %%

# %% [markdown]
#
# ## Eigene Bereinigungsfunktion übergeben
#
# - Verschiedene Quellen brauchen verschiedene Bereinigung
# - `clean_documents` akzeptiert eine beliebige Bereinigungsfunktion

# %%
def thorough_clean(text):
    text = ftfy.fix_text(text)
    text = clean_text_2(text)
    text = clean(text, no_urls=True, no_emails=True, no_emoji=True, to_ascii=False, lower=False)
    return text

# %%

# %%

# %%


# %% [markdown]
#
# ## Zusammenfassung: Text bereinigen
#
# - **Regex**: Eigene Bereinigungsfunktionen für spezifische Formate
# - **trafilatura**: Haupttext aus Webseiten extrahieren
# - **ftfy**: Kaputte Unicode-Zeichen und HTML-Entities reparieren
# - **clean-text**: Umfassende Normalisierung (URLs, Emojis, Whitespace, ...)
# - **Pipeline**: Bereinigungsfunktionen mit Document Loadern kombinieren
#
# **Nächster Schritt**: Text in Chunks aufteilen
