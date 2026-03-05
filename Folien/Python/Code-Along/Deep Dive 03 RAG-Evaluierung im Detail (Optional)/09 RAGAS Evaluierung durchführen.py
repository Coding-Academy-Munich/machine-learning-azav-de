# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>RAGAS Evaluierung durchführen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## RAGAS in der Praxis
#
# Jetzt wenden wir RAGAS auf unser RAG-System an:
# 1. Daten sammeln (haben wir schon!)
# 2. Metriken auswählen
# 3. Evaluierung durchführen
# 4. Ergebnisse interpretieren

# %%
import os
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall

# %%
load_dotenv()

# %% [markdown]
#
# ## Evaluator-LLM einrichten
#
# Wir erstellen einen separaten `AsyncOpenAI`-Client für die RAGAS-Metriken:

# %%
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
evaluator_llm = llm_factory(
    "mistralai/ministral-14b-2512",
    provider="openai",
    client=openai_client,
)
evaluator_embeddings = embedding_factory(
    "openai",
    model="openai/text-embedding-3-small",
    client=openai_client,
)

# %% [markdown]
#
# ## Metriken erstellen
#
# Jede Metrik bekommt das Evaluator-LLM (und ggf. Embeddings):

# %%
faithfulness = Faithfulness(llm=evaluator_llm)
answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
context_recall = ContextRecall(llm=evaluator_llm)

# %% [markdown]
#
# ## RAG-System aufbauen

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
    temperature=0,
)

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

documents = [Document(page_content=doc) for doc in docs_content]

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small",
)

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

vectorstore = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name="ml_docs_ragas_eval",
    location=":memory:",
    retrieval_mode=RetrievalMode.HYBRID,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %% [markdown]
#
# ## Daten sammeln

# %%
test_questions = [
    {
        "question": "Was ist Overfitting?",
        "reference": "Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt und nicht auf neue Daten generalisiert.",
    },
    {
        "question": "Wie werden Neural Networks trainiert?",
        "reference": "Neural Networks werden durch Backpropagation und Gradient Descent trainiert.",
    },
    {
        "question": "Was ist der Vorteil von Random Forests?",
        "reference": "Random Forests kombinieren viele Decision Trees und sind oft genauer als einzelne Modelle.",
    },
    {
        "question": "Wie hilft RAG bei LLMs?",
        "reference": "RAG hilft LLMs, präziser mit spezifischem Wissen zu antworten.",
    },
    {
        "question": "Was ist Linear Regression?",
        "reference": "Linear Regression ist eine Methode des überwachten Lernens, die lineare Beziehungen zwischen Features und Zielvariable modelliert.",
    },
]

# %%
eval_data = []
for item in test_questions:
    question = item["question"]
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    eval_data.append(
        {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": [doc.page_content for doc in docs],
            "reference": item["reference"],
        }
    )

# %%

# %% [markdown]
#
# ## Evaluierung durchführen
#
# Jede Metrik hat eine `.ascore()`-Methode für einzelne Beispiele
# und `.abatch_score()` für mehrere auf einmal:

# %%
faith_results = await faithfulness.abatch_score([
    {"user_input": d["user_input"], "response": d["response"], "retrieved_contexts": d["retrieved_contexts"]}
    for d in eval_data
])

# %%
relevancy_results = await answer_relevancy.abatch_score([
    {"user_input": d["user_input"], "response": d["response"]}
    for d in eval_data
])

# %%
recall_results = await context_recall.abatch_score([
    {"user_input": d["user_input"], "retrieved_contexts": d["retrieved_contexts"], "reference": d["reference"]}
    for d in eval_data
])

# %% [markdown]
#
# ## Ergebnisse interpretieren
#
# Der Gesamtscore:

# %%
print("Gesamtergebnisse / Overall Results:")
for name, results in [("Faithfulness", faith_results), ("Answer Relevancy", relevancy_results), ("Context Recall", recall_results)]:
    scores = [r.value for r in results]
    print(f"  {name:<25} {sum(scores)/len(scores):.2f}")

# %% [markdown]
#
# ## Ergebnisse pro Frage
#
# Der Gesamtscore versteckt Probleme bei einzelnen Fragen.
# Schauen wir uns jede Frage an:

# %%
df = pd.DataFrame({
    "user_input": [d["user_input"] for d in eval_data],
    "faithfulness": [r.value for r in faith_results],
    "answer_relevancy": [r.value for r in relevancy_results],
    "context_recall": [r.value for r in recall_results],
})

# %%

# %% [markdown]
#
# ## Was sagen uns die Ergebnisse?
#
# - **Faithfulness < 1.0**: Das LLM fügt eigenes Wissen hinzu → Prompt verbessern
# - **Answer Relevancy < 1.0**: Antworten gehen teilweise am Thema vorbei
# - **Context Recall < 1.0**: Retriever findet nicht alle relevanten Infos → k erhöhen?
#
# **Jeder niedrige Score zeigt uns, wo wir verbessern können!**

# %% [markdown]
#
# ## Kosten und Performance
#
# - 5 Beispiele × 3 Metriken = ~**15 LLM-Aufrufe** für die Evaluierung
# - Mit ministral-14b: **< $0.05** für den ganzen Durchlauf
# - Dauer: ~1-2 Minuten
#
# **Tipps zur Kostenoptimierung:**
# - Starten Sie immer mit wenigen Beispielen (5-10)
# - Erst wenn alles funktioniert: auf mehr Beispiele skalieren
# - Günstigere Modelle für Evaluierung sind oft ausreichend

# %% [markdown]
#
# ## Zusammenfassung
#
# - `abatch_score()` führt RAGAS-Metriken auf unseren Daten aus
# - Jede Metrik bekommt nur die Felder, die sie braucht
# - Drei Metriken messen unterschiedliche Qualitätsdimensionen
# - Niedrige Scores zeigen Verbesserungspotenzial:
#   - Faithfulness niedrig → Prompt anpassen
#   - Relevancy niedrig → System-Prompt überarbeiten
#   - Context Recall niedrig → Retriever verbessern (k, Chunking)
#
# **Nächster Schritt**: Im Workshop alles zusammen anwenden!
