# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Chains und LCEL - Einführung</b>
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
# translation = llm.invoke("Übersetze: " + summary.content)
# ```
#
# Das funktioniert, aber es ist **umständlich**.

# %% [markdown]
#
# In der Praxis bestehen die einzelnen Aktionen selber aus mehreren Schritten:
#
# ```python
# prompt = template.invoke({"input": "Was ist Python in einem Satz?"})
# response = llm.invoke(prompt)
# text = response.content
# ```

# %% [markdown]
#
# ## Bisheriger Ablauf
#
# 1. `prompt.invoke()` formatiert den Prompt.</br>Wir speichern das Ergebnis in einer Variablen.
# 2. `llm.invoke()` ruft das Modell auf.</br>Wir speichern die Antwort in einer Variablen.
# 3. `response.content` extrahiert den Text.
#
# Das funktioniert, aber ist umständlich.

# %%
import os
from dotenv import load_dotenv

# %%
load_dotenv()

# %%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# %%
model = "mistralai/ministral-14b-2512"

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model,
)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Die Lösung: LCEL
#
# **LCEL** = LangChain Expression Language
#
# - Templates und LLMs in LangChain sind **Runnables**
#   - Haben eine `invoke()`-Methode
#   - Runnables können mit `|` verkettet werden
#   - Ausgabe von Schritt 1 wird Eingabe von Schritt 2
# - Klar, lesbar, kombinierbar

# %% [markdown]
#
# ## Eine einfache Chain
#
# Die einfachste Chain: Template → LLM → Text
#
# Wir wissen:
# - Templates sind Runnables, die Eingaben formatieren
# - LLMs sind Runnables, die Prompts in Antworten umwandeln
# - Wir brauchen ein Runnable, das die Antwort in Text umwandelt<br>
#   → `StrOutputParser`

# %%
from langchain_core.output_parsers import StrOutputParser

# %%

# %%

# %%


# %%

# %%

# %%

# %%

# %% [markdown]
#
# Was passiert hier?
#
# 1. `template` formatiert die Eingabe
# 2. `|` leitet das Ergebnis weiter
# 3. `llm` ruft das Modell auf
# 4. `|` leitet das Ergebnis weiter
# 5. `output_parser` extrahiert den Text

# %% [markdown]
#
# ## Eine interaktive Oberfläche
#
# Wir können unsere Chain mit einer Benutzeroberfläche verbinden
#
# **Gradio** macht das einfach:
# - `gr.Interface()` erstellt eine einfache Web-Oberfläche
# - Eingabefeld → Funktion → Ausgabe
# - Perfekt für einfache Demos

# %% [markdown]
#
# ### Erklärungsbot
#
# Erstellen wir einen Bot, der Begriffe in einem Satz erklärt:

# %%
explanation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du bist ein hilfreicher Assistent. Antworte immer in genau einem Satz. "
         "Formatiere deine Antwort in Markdown."),
        ("human", "Was ist {input} in einem Satz?"),
    ]
)

# %%
explanation_chain = explanation_template | llm | output_parser

# %%
explanation_chain.invoke({"input": "Python"})


# %%
import gradio as gr

# %%

# %%

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
# ## Zusammenfassung
#
# - **LCEL**: Pipe-Operator `|` verbindet Runnables
# - **Einfache Chain**: `template | llm | output_parser`
# - **StrOutputParser**: Extrahiert Text aus LLM-Antwort
# - **LangSmith**: Zeigt jeden Schritt der Chain

# %% [markdown]
#
# ## Workshop: Haiku Generator
#
# **Ziel**: Erstellen Sie eine einfache Chain, die Haikus generiert!
#
# Ein Haiku ist ein japanisches Kurzgedicht:
# - Zeile 1: 5 Silben
# - Zeile 2: 7 Silben
# - Zeile 3: 5 Silben
#
# Sie werden:
# 1. Ein Template für Haiku-Generierung erstellen
# 2. Die Chain bauen: `template | llm | output_parser`
# 3. Haikus zu verschiedenen Themen generieren

# %% [markdown]
#
# ### Aufgabe 1: Template erstellen
#
# Erstellen Sie ein `ChatPromptTemplate` mit:
# - System-Nachricht: Erklärt dem LLM, dass es ein Haiku-Dichter ist
# - Human-Nachricht: Das Thema für das Haiku (als `{topic}` Platzhalter)

# %%
haiku_template = ...

# %% [markdown]
#
# ### Aufgabe 2: Chain erstellen
#
# Erstellen Sie die Chain mit dem Pipe-Operator:
# - Template → LLM → Output Parser

# %%
haiku_chain = ...

# %% [markdown]
#
# ### Aufgabe 3: Haikus generieren
#
# Testen Sie Ihre Chain mit verschiedenen Themen!

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 4: Gradio Interface (Optional)
#
# Erstellen Sie eine interaktive Benutzeroberfläche für den Haiku Generator.

# %%
import gradio as gr

# %%
def generate_haiku(topic):
    """Generiert ein Haiku zum gegebenen Thema."""
    # TODO: Implementieren Sie die Funktion
    pass

# %%


# %%

# %%

# %% [markdown]
#
# ### Überprüfen Sie LangSmith!
#
# Nach dem Testen sehen Sie in LangSmith:
#
# 1. **RunnableSequence**: Die Haiku-Chain
# 2. **ChatPromptTemplate**: Thema wird eingesetzt
# 3. **ChatOpenAI**: LLM generiert das Haiku
# 4. **StrOutputParser**: Text wird extrahiert
#
# Schauen Sie sich den Token-Verbrauch und die Latenz an!

# %% [markdown]
#
# ## Workshop Zusammenfassung
#
# Sie haben gelernt:
# - Einfache Chains mit `template | llm | output_parser` zu bauen
# - Templates mit Platzhaltern für variable Eingaben zu erstellen
# - Chains mit `invoke()` auszuführen
# - Optional: Gradio-Interfaces für interaktive Nutzung
