# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Chains und LCEL</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Das Problem: Mehrere Schritte
#
# Was wenn wir **mehrere Dinge** hintereinander tun wollen?
#
# **Beispiel**: Text zusammenfassen und dann übersetzen
#
# ```python
# # Schritt 1: Zusammenfassen
# summary = llm.invoke("Fasse zusammen: " + text)
#
# # Schritt 2: Übersetzen
# translation = llm.invoke("Übersetze: " + summary)
# ```
#
# Das funktioniert, aber es ist **umständlich**.

# %% [markdown]
#
# ## Die Lösung: LCEL
#
# **LCEL** = LangChain Expression Language
#
# - Komponenten mit `|` verketten (wie Unix-Pipes!)
# - Ausgabe von Schritt 1 wird Eingabe von Schritt 2
# - Klar, lesbar, komponierbar

# %%
import os
from dotenv import load_dotenv

load_dotenv()

# %%
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ml-azav-course"

# %%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# %%
model = "mistralai/ministral-14b-2512"

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model,
)

# %% [markdown]
#
# ## Eine einfache Chain
#
# Die einfachste Chain: Template → LLM → Text

# %%
template = ChatPromptTemplate.from_messages(
    [("system", "Du bist ein hilfreicher Assistent."), ("human", "{input}")]
)

# %%
chain = template | llm | StrOutputParser()

# %% [markdown]
#
# Was passiert hier?
#
# 1. `template` formatiert die Eingabe
# 2. `|` leitet das Ergebnis weiter
# 3. `llm` ruft das Modell auf
# 4. `|` leitet das Ergebnis weiter
# 5. `StrOutputParser()` extrahiert den Text

# %%

# %%

# %% [markdown]
#
# ## LangSmith zeigt die Chain
#
# In LangSmith sehen Sie jetzt:
#
# 1. **RunnableSequence**: Die gesamte Chain
# 2. **ChatPromptTemplate**: Template-Verarbeitung
# 3. **ChatOpenAI**: LLM-Aufruf
# 4. **StrOutputParser**: Text-Extraktion
#
# Jeder Schritt ist sichtbar!

# %% [markdown]
#
# ## Zwei Chains hintereinander
#
# Jetzt bauen wir eine Pipeline:
# 1. Text zusammenfassen
# 2. Zusammenfassung übersetzen

# %%
summarize_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du fasst Texte prägnant zusammen."),
        ("human", "Fasse diesen Text in 2-3 Sätzen zusammen:\n\n{text}"),
    ]
)

# %%
translate_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du übersetzt Texte ins {language}."),
        ("human", "Übersetze:\n\n{text}"),
    ]
)

# %% [markdown]
#
# ## Chains kombinieren
#
# Wir können die Ausgabe einer Chain als Eingabe für die nächste verwenden:

# %%
summarize_chain = summarize_template | llm | StrOutputParser()

# %%
sample_text = """
Python ist eine vielseitige, interpretierte Programmiersprache, die 1991 von
Guido van Rossum entwickelt wurde. Sie zeichnet sich durch ihre klare,
lesbare Syntax aus und unterstützt mehrere Programmierparadigmen wie
objektorientierte, imperative und funktionale Programmierung. Python wird
heute in vielen Bereichen eingesetzt, darunter Webentwicklung, Datenanalyse,
künstliche Intelligenz, wissenschaftliches Rechnen und Automatisierung.
Die große Standardbibliothek und die aktive Community machen Python zu
einer der beliebtesten Programmiersprachen weltweit.
"""

# %%

# %%

# %% [markdown]
#
# ## Jetzt übersetzen
#
# Die Zusammenfassung an die Übersetzungs-Chain weitergeben:

# %%
translate_chain = translate_template | llm | StrOutputParser()

# %%

# %%

# %% [markdown]
#
# ## Eine kombinierte Chain
#
# Mit `RunnablePassthrough` und `RunnableLambda` können wir alles verbinden:

# %%
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# %%
full_pipeline = (
    {"text": RunnablePassthrough()}
    | summarize_chain
    | (lambda summary: {"text": summary, "language": "Englisch"})
    | translate_chain
)

# %%

# %%

# %% [markdown]
#
# ## LangSmith: Die volle Pipeline
#
# In LangSmith sehen Sie jetzt die **gesamte Pipeline**:
#
# 1. Eingabe: Der Originaltext
# 2. Zusammenfassung: Erste LLM-Aufruf
# 3. Transformation: Text → Dict mit Sprache
# 4. Übersetzung: Zweiter LLM-Aufruf
# 5. Ausgabe: Der übersetzte Text
#
# **Debugging wird einfach**: Wenn etwas schiefgeht, sehen Sie genau wo!

# %% [markdown]
#
# ## RunnableParallel: Mehrere Dinge gleichzeitig
#
# Was wenn wir mehrere Informationen brauchen?

# %%
from langchain_core.runnables import RunnableParallel

# %%
sentiment_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Analysiere die Stimmung des Textes. Antworte nur mit: positiv, negativ oder neutral"),
        ("human", "{text}"),
    ]
)

keywords_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Extrahiere 3-5 Schlüsselwörter aus dem Text. Liste sie kommasepariert auf."),
        ("human", "{text}"),
    ]
)

# %%
sentiment_chain = sentiment_template | llm | StrOutputParser()
keywords_chain = keywords_template | llm | StrOutputParser()

# %%
analysis_chain = RunnableParallel(
    sentiment=sentiment_chain,
    keywords=keywords_chain,
    original=RunnablePassthrough(),
)

# %%

# %%

# %% [markdown]
#
# ## Das Ergebnis: Ein Dictionary
#
# `RunnableParallel` gibt ein Dictionary mit allen Ergebnissen zurück:

# %%

# %%

# %% [markdown]
#
# ## Wann Chains verwenden?
#
# **Chains sind gut für**:
# - Wiederholbare Pipelines
# - Multi-Step-Verarbeitung
# - Klare, dokumentierte Workflows
# - Debugging mit LangSmith
#
# **Einfaches `invoke()` reicht für**:
# - Einzelne Aufrufe
# - Einfache Chatbots
# - Prototyping

# %% [markdown]
#
# ## Zusammenfassung
#
# - **LCEL**: Pipe-Operator `|` für Chains
# - **StrOutputParser**: Text aus LLM-Antwort extrahieren
# - **RunnableParallel**: Mehrere Chains gleichzeitig
# - **RunnableLambda**: Custom Transformationen
# - **LangSmith**: Jeden Schritt der Chain debuggen

# %% [markdown]
#
# ## Workshop: Content Transformer
#
# **Ziel**: Eine Pipeline, die Text zusammenfasst, übersetzt und formatiert!
#
# **Aufgaben**:
# 1. Zusammenfassungs-Chain erstellen
# 2. Übersetzungs-Chain erstellen
# 3. Stil-Anpassungs-Chain erstellen (formal/casual)
# 4. Alles zu einer Pipeline kombinieren
# 5. Gradio Interface bauen

# %%
import gradio as gr

# %% [markdown]
#
# ### Teil 1: Die einzelnen Chains

# %%
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du fasst Texte prägnant zusammen. Maximal 3 Sätze."),
        ("human", "Fasse zusammen:\n\n{text}"),
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du übersetzt Texte ins {target_language}. Behalte den Stil bei."),
        ("human", "Übersetze:\n\n{text}"),
    ]
)

style_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du passt den Stil von Texten an. Zielstil: {style}"),
        ("human", "Passe den Stil an:\n\n{text}"),
    ]
)

# %%
summary_chain = summary_template | llm | StrOutputParser()
translation_chain = translation_template | llm | StrOutputParser()
style_chain = style_template | llm | StrOutputParser()

# %% [markdown]
#
# ### Teil 2: Die kombinierte Pipeline

# %%
def transform_content(text, target_language, style):
    """Transform content through the pipeline."""
    # TODO: Implement the pipeline
    # 1. Summarize the text
    # 2. Translate to target_language
    # 3. Adjust to target style
    pass

# %% [markdown]
#
# ### Teil 3: Testen

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ### Teil 4: Gradio Interface

# %%
# TODO: Create Gradio interface for the content transformer

# %%

# %% [markdown]
#
# ### Überprüfen Sie LangSmith!
#
# Nach dem Testen sehen Sie in LangSmith:
#
# 1. **Drei separate LLM-Aufrufe** für jeden Schritt
# 2. **Eingabe und Ausgabe** jedes Schritts
# 3. **Gesamtdauer** und Token-Verbrauch
# 4. Wenn etwas schiefgeht: **Genau wo der Fehler war**

# %% [markdown]
#
# ## Workshop-Aufgaben Zusammenfassung
#
# ### Basis (Pflicht):
# 1. Drei Chains erstellen (Zusammenfassen, Übersetzen, Stil)
# 2. Pipeline kombinieren
# 3. Gradio Interface bauen
# 4. LangSmith Traces überprüfen
#
# ### Erweitert (Optional):
# 5. Länge der Zusammenfassung einstellbar machen
# 6. Mehrere Sprachen gleichzeitig (mit RunnableParallel)
# 7. Fehlerbehandlung verbessern

# %%
