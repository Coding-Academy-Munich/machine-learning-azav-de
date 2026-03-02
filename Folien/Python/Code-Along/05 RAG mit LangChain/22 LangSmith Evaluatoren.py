# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LangSmith Evaluatoren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was ist ein Evaluator?
#
# Ein Evaluator ist eine **Funktion**, die eine Antwort bewertet
#
# ```
# Evaluator(inputs, outputs, reference_outputs) → Score
# ```
#
# Drei Typen:
# - **Heuristisch**: Einfache Regeln (Länge, Keywords)
# - **LLM-als-Richter**: Ein LLM bewertet die Antwort
# - **Custom**: Beliebige Python-Funktion

# %%
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# %%
load_dotenv()

# %% [markdown]
#
# ## Evaluator-Signatur
#
# Jeder LangSmith-Evaluator bekommt drei Argumente:
#
# - `inputs`: Die Eingaben (z.B. `{"question": "Was ist ...?"}`)
# - `outputs`: Was unser System geantwortet hat
# - `reference_outputs`: Die erwartete Antwort aus dem Dataset

# %% [markdown]
#
# ## Heuristischer Evaluator: Antwortlänge
#
# - Prüft, ob die Antwort eine sinnvolle Länge hat
# - Zu kurz → wahrscheinlich unvollständig
# - Zu lang → wahrscheinlich nicht präzise

# %%
def answer_length_ok(inputs, outputs, reference_outputs):
    answer = outputs.get("answer", "")
    length = len(answer)
    return {
        "key": "answer_length_ok",
        "score": 1 if 20 < length < 500 else 0,
        "comment": f"Answer length: {length} characters",
    }

# %% [markdown]
#
# - `key`: Name der Metrik (erscheint in LangSmith)
# - `score`: 0 oder 1 (oder Wert zwischen 0 und 1)
# - `comment`: Optionale Erklärung

# %% [markdown]
#
# ## LLM-als-Richter: Korrektheit
#
# - Ein LLM prüft, ob die generierte Antwort korrekt ist
# - Vergleicht mit der Referenz-Antwort
# - Nutzt **Structured Output** für klare Bewertung

# %%
from typing import TypedDict

# %%
class CorrectnessGrade(TypedDict):
    correct: bool
    explanation: str

# %%
judge_llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
    temperature=0,
)

# %%
grader = judge_llm.with_structured_output(CorrectnessGrade)

# %% [markdown]
#
# ## Der Korrektheit-Evaluator

# %%
CORRECTNESS_PROMPT = """\
You are a grading assistant. Compare the generated answer with the reference answer.

Question: {question}
Reference answer: {reference}
Generated answer: {answer}

Grade whether the generated answer is correct and consistent with the reference.
Minor wording differences are acceptable.
If the reference says the information is not available, the generated answer should also indicate that."""

# %%
def correctness(inputs, outputs, reference_outputs):
    question = inputs["question"]
    reference = reference_outputs["answer"]
    answer = outputs.get("answer", "")

    grade = grader.invoke(
        CORRECTNESS_PROMPT.format(
            question=question, reference=reference, answer=answer
        )
    )

    return {
        "key": "correctness",
        "score": 1 if grade["correct"] else 0,
        "comment": grade["explanation"],
    }

# %% [markdown]
#
# ## LLM-als-Richter: Groundedness
#
# - Prüft, ob die Antwort **nur** auf dem abgerufenen Kontext basiert
# - Erkennt Halluzinationen
# - Braucht den Kontext als zusätzliche Information

# %%
class GroundednessGrade(TypedDict):
    grounded: bool
    explanation: str

# %%
groundedness_grader = judge_llm.with_structured_output(GroundednessGrade)

# %%
GROUNDEDNESS_PROMPT = """\
You are a grading assistant. Check if the answer is grounded in (supported by) the provided context.

Question: {question}
Context: {context}
Answer: {answer}

The answer should only contain information that can be found in or reasonably inferred from the context.
If the answer contains claims not supported by the context, it is NOT grounded."""

# %%
def groundedness(inputs, outputs, reference_outputs):
    question = inputs["question"]
    context = outputs.get("contexts", "No context available")
    answer = outputs.get("answer", "")

    grade = groundedness_grader.invoke(
        GROUNDEDNESS_PROMPT.format(
            question=question, context=context, answer=answer
        )
    )

    return {
        "key": "groundedness",
        "score": 1 if grade["grounded"] else 0,
        "comment": grade["explanation"],
    }

# %% [markdown]
#
# ## Evaluatoren testen
#
# Bevor wir die Evaluatoren auf das ganze Dataset anwenden, testen wir sie:

# %%
test_inputs = {"question": "Was ist Overfitting?"}

# %%
test_outputs = {
    "answer": "Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt.",
    "contexts": "Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt. Regularisierung hilft dagegen.",
}

# %%
test_reference = {
    "answer": "Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt und nicht auf neue Daten generalisiert."
}

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Token-Kosten beachten
#
# - Jeder LLM-als-Richter-Aufruf kostet Token
# - 6 Beispiele × 2 LLM-Evaluatoren = **12 LLM-Aufrufe**
# - Mit ministral-14b via OpenRouter: sehr günstig (~$0.001/Aufruf)
# - **Tipp**: Während der Entwicklung mit wenigen Beispielen testen

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Heuristische Evaluatoren**: Schnell, kostenlos, aber begrenzt
# - **LLM-als-Richter**: Flexibel und nuanciert
#   - Korrektheit: Vergleicht mit Referenz-Antwort
#   - Groundedness: Prüft auf Halluzinationen
# - Structured Output (`TypedDict`) für klare Bewertungen
# - Immer erst einzeln testen, dann auf Dataset anwenden
#
# **Nächster Schritt**: Experimente durchführen!
