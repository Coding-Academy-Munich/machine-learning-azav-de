# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Parallele Ausführung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import os
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# %%
model = "mistralai/ministral-14b-2512"

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model,
)

# %%
sample_text = """\
Die künstliche Intelligenz hat in den letzten Jahren enorme Fortschritte
gemacht. Besonders im Bereich der natürlichen Sprachverarbeitung haben große
Sprachmodelle wie GPT und Claude gezeigt, dass Maschinen menschenähnliche Texte
generieren können. Diese Entwicklung hat weitreichende Auswirkungen auf viele
Branchen, von der Softwareentwicklung bis zum Kundenservice.

Allerdings bringt dieser technologische Sprung auch erhebliche Herausforderungen
mit sich, die die Begeisterung für diese Fähigkeiten dämpfen. Während große
Sprachmodelle in der Mustererkennung und der flüssigen Textgenerierung
brillieren, neigen sie dazu, "Halluzinationen" zu erzeugen - also plausibel
klingende, aber faktisch falsche Informationen - und können unbeabsichtigt
Vorurteile verstärken, die in ihren Trainingsdaten vorhanden sind.
Organisationen, die diese Systeme einsetzen, müssen daher stark in
Verifikationspipelines, Mechanismen zur menschlichen Aufsicht und Programme zur
Umschulung der Belegschaft investieren, um sicherzustellen, dass KI die
kritische menschliche Urteilsfähigkeit ergänzt und nicht ersetzt, insbesondere
in Bereichen mit hohen Einsätzen wie Gesundheitswesen, Rechtsdienstleistungen
und Finanzberatung.

Mit Blick auf die Zukunft deutet die Entwicklung auf immer ausgefeiltere
multimodale Systeme hin, die Text, Code, Bilder und strukturierte Daten
integrieren, um komplexe, domänenübergreifende Probleme zu lösen. Doch das Tempo
der Innovation wirft dringende Fragen zur Governance auf: Politiker haben
Schwierigkeiten, mit Fähigkeiten Schritt zu halten, die sich monatlich anstatt
jährlich weiterentwickeln, während Pädagogen damit kämpfen, Schüler auf einen
Arbeitsmarkt vorzubereiten, in dem die Zusammenarbeit zwischen Mensch und KI zum
Standardarbeitsmodus wird. Das wahre Maß für den Erfolg dieser Technologie wird
wahrscheinlich nicht von ihren rohen Fähigkeiten abhängen, sondern von unserer
Fähigkeit, sie innerhalb ethischer Rahmenbedingungen einzusetzen, die
Transparenz, Verantwortlichkeit und einen gerechten Zugang zu ihren Vorteilen
priorisieren.
"""

# %% [markdown]
#
# ## RunnableParallel: Mehrere Dinge gleichzeitig
#
# Was wenn wir mehrere Informationen brauchen?

# %%
from langchain_core.runnables import RunnableParallel

# %%
sentiment_prompt = "Analysiere die Stimmung des Textes. Antworte nur mit: positiv, negativ oder neutral"

# %%
sentiment_template = ChatPromptTemplate.from_messages(
    [("system", sentiment_prompt), ("human", "{text}")]
)

# %%
keywords_prompt = """\
Extrahiere 3-5 Schlüsselwörter aus dem Text.
Liste sie kommasepariert auf."""

# %%
keywords_template = ChatPromptTemplate.from_messages(
    [("system", keywords_prompt), ("human", "{text}")]
)

# %%
sentiment_chain = sentiment_template | llm | StrOutputParser()

# %%
keywords_chain = keywords_template | llm | StrOutputParser()

# %% [markdown]
#
# ## Das Problem: Mehrere unabhängige Analysen
#
# Wir wollen **mehrere Informationen** aus demselben Text extrahieren:
#
# - Stimmung (positiv/negativ/neutral)
# - Schlüsselwörter
#
# Diese Analysen sind **unabhängig voneinander** - keine braucht das Ergebnis
# der anderen.

# %% [markdown]
#
# ### Ansatz 1: Mit Chaining?
#
# Könnten wir die Chains einfach verketten?

# %%
awkward_chain = (
    sentiment_chain
    | RunnableLambda(lambda _: sample_text)
    | keywords_chain
)


# %%

# %% [markdown]
#
# **Problem**: Wir verlieren das Sentiment-Ergebnis!
#
# Das Chaining gibt nur das letzte Ergebnis zurück.

# %% [markdown]
#
# ### Ansatz 2: Sequentielle Funktion
#
# Wir können eine Funktion schreiben, die beide Chains nacheinander aufruft:

# %%
def analyze_sequentially(text):
    """Analyze text sequentially - works but slow."""
    sentiment = sentiment_chain.invoke(text)
    keywords = keywords_chain.invoke(text)
    return {"sentiment": sentiment, "keywords": keywords}

# %%

# %%

# %% [markdown]
#
# ### Probleme mit dem sequentiellen Ansatz
#
# - **Umständlich**: Manuelle Orchestrierung für jede Chain
# - **Langsam**: Wir warten auf Sentiment, bevor Keywords startet
# - **Skaliert schlecht**: Mehr Chains = mehr manueller Code

# %% [markdown]
#
# ## Die Lösung: RunnableParallel
#
# `RunnableParallel` löst beide Probleme:
#
# - Alle Ergebnisse werden gesammelt
# - Chains laufen gleichzeitig (parallel)

# %%

# %%

# %%

# %% [markdown]
#
# ### Zeitvergleich: Sequentiell vs. Parallel
#
# Messen wir den Unterschied:

# %%
import time

# %%
num_runs = 5

# %%
start = time.time()
for _ in range(num_runs):
    result_seq = analyze_sequentially(sample_text)
sequential_time = time.time() - start

# %%
start = time.time()
for _ in range(num_runs):
    result_par = analysis_chain.invoke({"text": sample_text})
parallel_time = time.time() - start

# %%

# %% [markdown]
#
# ## RunnablePassthrough: Eingabe durchreichen
#
# - Gibt die Eingabe unverändert zurück
# - Nützlich in `RunnableParallel`, um Originalwerte zu behalten

# %%
from langchain_core.runnables import RunnablePassthrough

# %%

# %%

# %%

# %% [markdown]
#
# ### RunnablePassthrough in RunnableParallel
#
# Mit `RunnablePassthrough` können wir die Originaleingabe behalten:

# %%

# %%

# %%

# %% [markdown]
#
# ## Dictionary-Kurzform
#
# Ein Dictionary in einer Chain wird automatisch zu `RunnableParallel`:

# %% [markdown]
#
# ### Äquivalenz
#
# Diese beiden sind äquivalent:
#
# ```python
# # Explizit
# RunnableParallel(sentiment=sentiment_chain, keywords=keywords_chain)
#
# # Kurzform (nur in einer Chain mit |)
# some_runnable | {"sentiment": sentiment_chain, "keywords": keywords_chain}
# ```

# %%

# %%

# %%

# %% [markdown]
#
# ## Das Muster verstehen!
#
# Jetzt verstehen wir `{"text": RunnablePassthrough()}`:
#
# 1. Dict in Chain → wird zu `RunnableParallel`
# 2. `RunnablePassthrough()` → gibt Eingabe unverändert zurück
# 3. Ergebnis: `{"text": <ursprüngliche Eingabe>}`

# %% [markdown]
#
# ### String zu Dict für Templates
#
# Dieses Muster wandelt einen String in ein Dictionary um:

# %%
rest_of_chain = RunnablePassthrough()

# %%

# %%

# %% [markdown]
#
# ### Wichtig!
#
# - Das funktioniert nur in einer Chain
# - Ohne ein `Runnable` davor oder danach ergibt das einen Fehler:

# %%
# bad_chain = {"text": RunnablePassthrough()}

# %%


# %% [markdown]
#
# ### Alternative: Lambda
#
# Für einfache Fälle ist ein Lambda oft klarer:

# %%
summarize_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du fasst Texte prägnant zusammen."),
        ("human", "Fasse diesen Text zusammen:\n\n{text}"),
    ]
)

# %%
# Beide Varianten funktionieren - wählen Sie, was klarer ist:

# Idiomatisches LangChain
chain1 = {"text": RunnablePassthrough()} | summarize_template | llm

# Expliziter mit Lambda
chain2 = (lambda x: {"text": x}) | summarize_template | llm

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
# - **Sequentiell vs. Parallel**: Unabhängige Chains profitieren von Parallelisierung
# - **RunnableParallel**: Mehrere Chains gleichzeitig (oder Dict-Kurzform)
# - **RunnablePassthrough**: Eingabe unverändert durchreichen
# - **Timing**: Parallele Ausführung kann signifikante Zeitersparnis bringen

# %% [markdown]
#
# ## Workshop: Multi-Language Translator
#
# **Ziel**: Nutzen Sie `RunnableParallel`, um Text gleichzeitig in mehrere
# Sprachen zu übersetzen!
#
# Sie werden:
# 1. Einzelne Übersetzungs-Chains für verschiedene Sprachen erstellen
# 2. Mit `RunnableParallel` alle Übersetzungen gleichzeitig ausführen
# 3. Die Ergebnisse vergleichen
# 4. Optional: Ein Gradio-Interface erstellen

# %% [markdown]
#
# ### Aufgabe 1: Übersetzungs-Templates erstellen
#
# Erstellen Sie Templates für Übersetzungen in verschiedene Sprachen:
# - Deutsch → Französisch
# - Deutsch → Spanisch
# - Deutsch → Italienisch

# %%
french_template = ...
spanish_template = ...
italian_template = ...

# %% [markdown]
#
# ### Aufgabe 2: Einzelne Chains erstellen
#
# Erstellen Sie für jede Sprache eine Chain: `template | llm | output_parser`

# %%
french_chain = ...
spanish_chain = ...
italian_chain = ...

# %% [markdown]
#
# ### Aufgabe 3: Parallele Chain erstellen
#
# Kombinieren Sie alle Chains mit `RunnableParallel`:

# %%
multi_translator = ...

# %% [markdown]
#
# ### Aufgabe 4: Testen
#
# Testen Sie den Multi-Language Translator mit einem Beispieltext:

# %%
test_sentence = "Künstliche Intelligenz verändert die Art, wie wir arbeiten und leben."

# %%

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 5: Gradio Interface (Optional)
#
# Erstellen Sie eine interaktive Benutzeroberfläche für den Multi-Language
# Translator.

# %%
import gradio as gr

# %%
def translate_to_all(text):
    """Übersetzt Text in alle Sprachen."""
    # TODO: Implementieren Sie die Funktion
    pass


translator_demo = gr.Interface(
    fn=translate_to_all,
    inputs=gr.Textbox(lines=3, label="Text", placeholder="Text zum Übersetzen..."),
    outputs=[
        gr.Textbox(label="Französisch"),
        gr.Textbox(label="Spanisch"),
        gr.Textbox(label="Italienisch"),
    ],
    title="Multi-Language Translator",
    description="Übersetzen Sie Text gleichzeitig in mehrere Sprachen!",
)

# %%

# %% [markdown]
#
# ### Überprüfen Sie LangSmith!
#
# Nach dem Testen sehen Sie in LangSmith:
#
# 1. **RunnableParallel**: Alle drei Übersetzungen
# 2. **Parallele Ausführung**: Die Chains laufen gleichzeitig
# 3. **Zeitersparnis**: Vergleichen Sie mit sequentieller Ausführung
#
# Die parallele Ausführung ist besonders nützlich bei mehreren unabhängigen
# LLM-Aufrufen!

# %% [markdown]
#
# ## Workshop Zusammenfassung
#
# ### Was Sie gelernt haben:
# 1. **RunnableParallel** für gleichzeitige Ausführung mehrerer Chains
# 2. **Effizienzgewinn** durch parallele LLM-Aufrufe
# 3. **Praktischer Anwendungsfall**: Multi-Language Translation
#
# ### Bonus-Aufgaben:
# - Fügen Sie weitere Sprachen hinzu (z.B. Portugiesisch, Niederländisch)
# - Messen Sie die Zeitersparnis gegenüber sequentieller Ausführung
# - Kombinieren Sie mit Zusammenfassung: Erst zusammenfassen, dann parallel
#   übersetzen
