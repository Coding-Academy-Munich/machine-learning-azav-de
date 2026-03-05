# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LangSmith Experimente</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was ist ein Experiment?
#
# Ein **Experiment** in LangSmith:
# 1. Nimmt ein Dataset (unsere Test-Fragen)
# 2. Führt unser System auf jeder Frage aus
# 3. Bewertet die Ergebnisse mit Evaluatoren
# 4. Speichert alles zur Analyse
#
# Wir können **mehrere Experimente vergleichen**!

# %%
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from typing import TypedDict

# %%
load_dotenv()

# %%
client = Client()

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
    temperature=0,
)

# %%
docs_content = [
    """Linear Regression ist eine Methode des überwachten Lernens.
    Sie modelliert lineare Beziehungen zwischen Features und Zielvariable.
    Die Normalgleichung kann verwendet werden, um optimale Parameter zu finden.
    Mean Squared Error (MSE) wird als Loss-Funktion verwendet.""",
    """Neural Networks bestehen aus Schichten von Neuronen.
    Jedes Neuron berechnet eine gewichtete Summe und wendet eine Aktivierungsfunktion an.
    Training erfolgt durch Backpropagation und Gradient Descent.
    Typische Aktivierungsfunktionen sind ReLU und Sigmoid.""",
    """Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt.
    Regularisierung und Cross-Validation helfen, Overfitting zu vermeiden.
    Ein gutes Modell generalisiert auf neue, ungesehene Daten.
    Dropout ist eine weitere Technik gegen Overfitting bei Neural Networks.""",
    """Large Language Models sind sehr große neuronale Netze.
    Sie werden auf Milliarden von Wörtern trainiert.
    RAG hilft LLMs, präziser mit spezifischem Wissen zu antworten.
    Prompt Engineering ist wichtig für gute Ergebnisse mit LLMs.""",
    """Decision Trees teilen Daten anhand von Entscheidungsregeln auf.
    Random Forests kombinieren viele Decision Trees für bessere Ergebnisse.
    Feature Importance zeigt, welche Features am wichtigsten sind.
    Ensemble-Methoden sind oft genauer als einzelne Modelle.""",
]

# %%
documents = [Document(page_content=doc) for doc in docs_content]

# %%
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small",
)

# %%
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# %%
vectorstore = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name="ml_docs_experiments",
    location=":memory:",
    retrieval_mode=RetrievalMode.HYBRID,
)

# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# %%
system_prompt = (
    "Du bist ein hilfreicher Assistent. Nutze den folgenden Kontext, um die Frage zu beantworten. "
    "Wenn der Kontext keine relevanten Informationen enthält, sage: "
    "'Diese Information ist nicht in meinen Dokumenten enthalten.' "
    "Halte die Antwort präzise.\n\n"
    "You are a helpful assistant. Use the following context to answer the question. "
    "If the context doesn't contain relevant information, say: "
    "'This information is not in my documents.' "
    "Keep the answer concise.\n\n"
    "Context: {context}"
)

# %%
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# %%
retriever_k2 = vectorstore.as_retriever(search_kwargs={"k": 2})

# %%
rag_chain_k2 = (
    {"context": retriever_k2 | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# %% [markdown]
#
# ## Die Target-Funktion
#
# LangSmith braucht eine **Target-Funktion**:
# - Bekommt die `inputs` aus dem Dataset
# - Führt unser RAG-System aus
# - Gibt die Ergebnisse als Dict zurück

# %%

# %% [markdown]
#
# - Die Funktion gibt `answer` und `contexts` zurück
# - `contexts` brauchen wir für den Groundedness-Evaluator

# %% [markdown]
#
# ## Evaluatoren definieren
#
# Die gleichen Evaluatoren wie zuvor:

# %%
def answer_length_ok(inputs, outputs, reference_outputs):
    answer = outputs.get("answer", "")
    length = len(answer)
    return {
        "key": "answer_length_ok",
        "score": 1 if 20 < length < 500 else 0,
        "comment": f"Answer length: {length} characters",
    }

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

# %% [markdown]
#
# ## Experiment 1: Baseline (k=2)
#
# Unser erstes Experiment mit dem aktuellen System:

# %%
dataset_name = "RAG Eval - ML Kurs"

# %%

# %% [markdown]
#
# ## Ergebnisse anzeigen

# %%
experiment_results_k2


# %% [markdown]
#
# ## Experiment 2: Mehr Kontext (k=4)
#
# Wird es besser, wenn wir mehr Dokumente abrufen?

# %%
retriever_k4 = vectorstore.as_retriever(search_kwargs={"k": 4})

# %%
rag_chain_k4 = (
    {"context": retriever_k4 | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# %%
def rag_target_k4(inputs):
    question = inputs["question"]
    docs = retriever_k4.invoke(question)
    answer = rag_chain_k4.invoke(question)
    return {
        "answer": answer,
        "contexts": "\n\n".join(doc.page_content for doc in docs),
    }

# %%
experiment_results_k4 = client.evaluate(
    rag_target_k4,
    data=dataset_name,
    evaluators=[answer_length_ok, correctness],
    experiment_prefix="more-context-k4",
)

# %%

# %% [markdown]
#
# ## Vergleich in LangSmith
#
# Öffnen Sie https://smith.langchain.com:
#
# 1. Gehen Sie zu **Datasets** → "RAG Eval - ML Kurs"
# 2. Sehen Sie beide Experimente nebeneinander
# 3. Vergleichen Sie die Scores pro Frage
#
# Die Web-Oberfläche zeigt:
# - Welche Fragen sich verbessert haben
# - Welche sich verschlechtert haben
# - Durchschnittliche Scores pro Evaluator

# %% [markdown]
#
# ## Der Evaluierungs-Zyklus
#
# ```
# 1. Baseline-Experiment durchführen
#        ↓
# 2. Ergebnisse analysieren
#        ↓
# 3. System verbessern (k, Prompt, Chunking...)
#        ↓
# 4. Neues Experiment durchführen
#        ↓
# 5. Vergleichen → Zurück zu 2.
# ```
#
# Dieser Zyklus ist der Kern von datengetriebener RAG-Verbesserung!

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Target-Funktion**: Wrapper um unser RAG-System
# - **`client.evaluate()`**: Führt ein Experiment durch
#   - Testet auf jedem Dataset-Beispiel
#   - Wendet alle Evaluatoren an
# - **Experimente vergleichen**: Sehen, welche Änderung hilft
# - **LangSmith UI**: Visuelle Analyse und Vergleich
#
# **Nächster Schritt**: RAGAS für spezialisierte RAG-Metriken!
