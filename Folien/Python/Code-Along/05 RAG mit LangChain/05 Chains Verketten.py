# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Chains verketten</b>
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
# ## Mehrere Verarbeitungsschritte
#
# Wir haben einen langen Text, den wir folgendermaßen verarbeiten wollen:
#
# - Zusammenfassen des Textes
# - Übersetzen der Zusammenfassung ins Englische
#
# Für jeden Schritt erstellen wir eine eigene Chain.

# %%
sample_text = """\
Python ist eine vielseitige, interpretierte Programmiersprache, die 1991 von
Guido van Rossum entwickelt wurde. Sie zeichnet sich durch ihre klare, lesbare
Syntax aus und unterstützt mehrere Programmiierparadigmen wie objektorientierte,
imperative und funktionale Programmierung. Python wird heute in vielen Bereichen
eingesetzt, darunter Webentwicklung, Datenanalyse, künstliche Intelligenz,
wissenschaftliches Rechnen und Automatisierung. Die große Standardbibliothek und
die aktive Community machen Python zu einer der beliebtesten Programmiersprachen
weltweit.

Besonders hervorzuheben ist die Einfachheit der Sprache, die es auch Anfängern
ermöglicht, schnell produktiven Code zu schreiben. Die Einrückung als
syntaktisches Element erzwingt zudem eine übersichtliche Code-Struktur. Mit
Frameworks wie Django und Flask lässt sich robuste Websoftware entwickeln,
während Bibliotheken wie NumPy, Pandas und TensorFlow Python zur bevorzugten
Wahl für Data Science und Machine Learning machen. Die Plattformunabhängigkeit
erlaubt zudem den Einsatz auf verschiedensten Betriebssystemen, von Windows über
macOS bis hin zu Linux und eingebetteten Systemen. Durch den Einsatz von
CPython, PyPy oder Cython kann bei Bedarf auch die Ausführungsgeschwindigkeit
deutlich gesteigert werden, was Python für leistungskritische Anwendungen
tauglich macht.

In jüngster Zeit hat Python insbesondere im Bereich der künstlichen Intelligenz
und des Machine Learnings vermehrt an Bedeutung gewonnen. Bibliotheken wie
TensorFlow, PyTorch und Scikit-learn bieten leistungsstarke Werkzeuge zum
Erstellen und Bereitstellen von Machine-Learning-Modellen. Die einfache
Handhabung und das umfangreiche Ökosystem haben Python zur bevorzugten Sprache
für Forscher und Entwickler gemacht, die an zukunftsweisenden KI-Anwendungen
arbeiten.
"""

# %%
summarize_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du fasst Texte prägnant zusammen."),
        ("human", "Fasse diesen Text in 2-3 Sätzen zusammen:\n\n{text}"),
    ]
)

# %%

# %%


# %%

# %%

# %%

# %% [markdown]
#
# ## Text übersetzen
#
# Wir erstellen eine zweite Chain für die Übersetzung eines Textes ins Englische.

# %%
simple_translate_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du übersetzt Texte ins Englische."),
        ("human", "Übersetze:\n\n{text}"),
    ]
)

# %%

# %%

# %%


# %% [markdown]
#
# ## Chains verketten
#
# Wir haben eine Chain für jeden Verarbeitungsschritt.
#
# Jetzt bauen wir daraus eine Pipeline:
#
# 1. Text zusammenfassen
# 2. Zusammenfassung übersetzen
#
# Dazu können wir einfach die Chains hintereinander schalten: Die Ausgabe der
# ersten Chain wird Eingabe der zweiten.

# %% [markdown]
#
# ## Einfaches Verketten
#
# Beide Chains haben die gleiche Signatur:
#
# - Eingabe: String
# - Ausgabe: String
#
# → Wir können sie direkt mit `|` verbinden!

# %%

# %%

# %%

# %% [markdown]
#
# ## Variable Zielsprache
#
# Was wenn wir in **verschiedene Sprachen** übersetzen wollen?
#
# - Französisch, Spanisch, Italienisch...
# - Die Sprache sollte ein **Parameter** sein!

# %%
translate_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du übersetzt Texte ins {language}."),
        ("human", "Übersetze:\n\n{text}"),
    ]
)


# %%

# %%

# %%

# %% [markdown]
#
# ## Das Problem: Signatur-Mismatch
#
# Jetzt können wir nicht mehr einfach verketten:
#
# ```python
# pipeline = summarize_chain | translate_chain  # Funktioniert **nicht**!
# ```
#
# Warum nicht?
#
# - `summarize_chain` gibt einen **String** zurück
# - `translate_chain` erwartet ein **Dict** mit `text` **und** `language`
#
# Wir brauchen einen Adapter!

# %% [markdown]
#
# ## Einschub: Anonyme Funktionen (Lambdas)
#
# - Eine **anonyme Funktion** (Lambda) ist eine Funktion ohne Namen
# - Nützlich, wenn wir eine kurze Funktion nur einmal brauchen

# %%

# %%

# %%

# %%

# %%
def process_greeting(func):
    return func("Hello world!")

# %%

# %%

# %%

# %%


# %% [markdown]
#
# ## RunnableLambda: Eigene Transformationen
#
# - Wir können beliebige Funktionen als Runnable verwenden
# - `RunnableLambda` macht eine Funktion zu einem Runnable

# %%
from langchain_core.runnables import RunnableLambda

# %%

# %%

# %% [markdown]
#
# ### Dictionaries mit RunnableLambda erstellen
#
# Wir können ein Lambda verwenden, um einen String in ein Dictionary umzuwandeln:

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ### Eine kombinierte Chain (explizit)
#
# Jetzt können wir unsere Chains verbinden:
#
# 1. Text in Dictionary umwandeln (mit Lambda)
# 2. Zusammenfassen
# 3. Ergebnis für Übersetzung vorbereiten (mit Lambda)
# 4. Übersetzen

# %%

# %%

# %%

# %% [markdown]
#
# ## LangSmith: Die volle Pipeline
#
# In LangSmith sehen Sie jetzt die **gesamte Pipeline**:
#
# 1. Eingabe: Der Originaltext
# 2. Transformation: String → Dict
# 3. Zusammenfassung: Erster LLM-Aufruf
# 4. Transformation: Text → Dict mit Sprache
# 5. Übersetzung: Zweiter LLM-Aufruf
#
# **Debugging wird einfach**: Wenn etwas schiefgeht, sehen Sie genau wo!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Chains verketten**: Ausgabe einer Chain → Eingabe der nächsten
# - **Einfaches Verketten**: Funktioniert wenn Signaturen passen
# - **Signatur-Mismatch**: Lösen mit `RunnableLambda`
# - **Lambda-Funktionen**: Anonyme Funktionen für kurze Transformationen
# - **LangSmith**: Zeigt die gesamte Pipeline zum Debugging

# %% [markdown]
#
# ## Einfache vs. konfigurierbare Chains
#
# **Einfache Chains** (String → String):
#
# - Jede Chain nimmt einen String und gibt einen String zurück
# - Lassen sich direkt mit `|` verketten
#
# **Konfigurierbare Chains** (mehrere Parameter):
#
# - Brauchen zusätzliche Parameter (z.B. Zielsprache)
# - Können nicht direkt verkettet werden
# - Brauchen Adapter (`RunnableLambda`) oder einen prozeduralen Ansatz

# %% [markdown]
#
# ## Workshop: Chain Komposition
#
# In diesem Workshop erstellen Sie eine Pipeline, die Text zusammenfasst, die
# Zusammenfassung übersetzt und die übersetzte Zusammenfassung in einen
# freundlichen Stil umwandelt.
#
# In einem zweiten Schritt geben Sie den Chains Parameter mit, um die
# Zielsprache und den Stil zu konfigurieren. Das wird die Verkettung schwieriger
# machen.
#
# Abschließend erstellen Sie ein Gradio-Interface für Ihre Pipeline.

# %%
import gradio as gr
from langchain_core.runnables import RunnablePassthrough

# %% [markdown]
#
# ### Teil 1: Einfache Chains (String → String)
#
# Erstellen Sie Chains mit **festem Verhalten**:
# - Zusammenfassen eines Textes
# - Einen Text ins Englische übersetzen
# - Einen Text in einen lockeren Stil umwandeln

# %% [markdown]
#
# #### Chain um einen Text zusammenzufassen
#
# - Erstellen Sie ein Prompt-Template für eine kompakte Zusammenfassung (max. 2
#   Sätze)
# - Erstellen Sie eine Chain, die das Template, das LLM und einen
#   `StrOutputParser` kombiniert

# %%
simple_summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du fasst Texte prägnant zusammen. Maximal 2 Sätze."),
        ("human", "{text}"),
    ]
)

# %%
simple_summary_chain = ...

# %% [markdown]
#
# #### Chain um einen Text zu übersetzen
#
# Erstellen Sie eine Chain, die einen Text ins Englische übersetzt. Folgen Sie
# dem gleichen Muster wie zuvor.

# %%
simple_translate_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du übersetzt Texte ins Englische."),
        ("human", "{text}"),
    ]
)

# %%
simple_translate_chain = ...

# %% [markdown]
#
# #### Chain um einen Text in einen lockeren Stil umzuwandeln
#
# Erstellen Sie eine Chain, die einen Text in einen lockeren, freundlichen Stil
# umwandelt. Folgen Sie dem gleichen Muster wie zuvor.

# %%
simple_style_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du schreibst Texte in einem lockeren, freundlichen Stil um."),
        ("human", "{text}"),
    ]
)

# %%
simple_style_chain = ...

# %% [markdown]
#
# ### Verketten der Chains
#
# Verketten Sie die drei Chains mit `|` zu einer Pipeline und testen Sie diese
# mit dem Beispieltext.

# %%
simple_pipeline = ...

# %%
test_text = """\
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

# %%

# %%

# %% [markdown]
#
# ### Teil 2: Konfigurierbare Chains
#
# Erstellen Sie konfigurierbare Versionen der Chains:
#
# - Die Zielsprache soll ein Parameter `{target_language}` sein
# - Der Stil soll ein Parameter `{style}` sein
#
# *Hinweis:* Überlegen Sie, was passiert wenn Sie diese Chains mit `|` verketten.

# %% [markdown]
#
# ### Erstellen Sie konfigurierbare Templates
#
# Erstellen Sie Templates mit Platzhaltern für Sprache und Stil:

# %%
summary_template = ...

# %%
summary_chain = ...


# %%
translation_template = ...

# %%
translation_chain = ...

# %%
style_template = ...

# %%
style_chain = ...


# %% [markdown]
#
# ### Teil 3: Prozedurale Lösung
#
# Die konfigurierbaren Chains lassen sich nicht direkt mit `|` verketten.
#
# Schreiben Sie eine Funktion `transform_content(text, target_language, style)`,
# die jede Chain explizit aufruft und die benötigten Parameter übergibt.
#
# Die Funktion soll ein Dictionary mit den Schlüsseln `summary`, `translation`
# und `final` zurückgeben.

# %%
def transform_content(text, target_language, style):
    """Transform content through the pipeline."""
    # TODO: Implement:
    # 1. Summarize the text
    # 2. Translate to target_language
    # 3. Apply style
    # Return dict with 'summary', 'translation', 'final'
    pass


# %% [markdown]
#
# Verwenden Sie `transform_content()`, um den Beispieltext:
#
# - ins Englische zu übersetzen und in einen lockeren, freundlichen Stil zu
#   bringen
# - ins Französische zu übersetzen und in einen poetischen Stil zu bringen
#
# (Sie können selbstverständlich auch andere Sprachen und Stile ausprobieren.)

# %%

# %%
print("=== Summary ===")

# %%

# %%
print("=== Translation ===")

# %%

# %%
print("=== Styled ===")

# %%

# %%

# %%
print("=== Summary ===")

# %%

# %%
print("=== Translation ===")

# %%

# %%
print("=== Styled ===")

# %%

# %% [markdown]
#
# ### Teil 4: Gradio Interface
#
# Erstellen Sie ein Gradio-Interface für Ihre `transform_content`-Funktion.
#
# - Verwenden Sie Dropdown-Menüs für Sprache und Stil
# - Zeigen Sie alle drei Ergebnisse (Zusammenfassung, Übersetzung, Finale Version)

# %%
LANGUAGES = ["Englisch", "Französisch", "Spanisch", "Italienisch"]
STYLES = ["formal und professionell", "casual und freundlich", "kurz und prägnant", "poetisch"]

# %%

# %%

# %%

# %% [markdown]
#
# *Tipp: Schauen Sie in LangSmith nach, um die einzelnen Schritte Ihrer Pipeline
# zu sehen.*
