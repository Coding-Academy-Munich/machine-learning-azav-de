# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>RAGAS: RAG-Evaluierung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was ist RAGAS?
#
# **RAGAS** = **R**etrieval **A**ugmented **G**eneration **A**ssessment
#
# - Open-Source Framework speziell für RAG-Evaluierung
# - Bietet standardisierte Metriken
# - Nutzt LLM-als-Richter unter der Haube
#
# **"LangSmith gibt uns den Workflow, RAGAS gibt uns die Metriken"**

# %%
# ! pip install ragas

# %%
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# %%
load_dotenv()

# %% [markdown]
#
# ## RAGAS mit OpenRouter einrichten
#
# RAGAS braucht ein LLM für die Bewertung.
# Wir nutzen den gleichen OpenRouter-Zugang:

# %%
from openai import AsyncOpenAI
from ragas.llms import llm_factory

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

# %% [markdown]
#
# - `llm_factory` erstellt ein RAGAS-kompatibles LLM aus einem OpenAI-Client
# - `AsyncOpenAI` wird benötigt — RAGAS-Metriken nutzen intern Async
# - Wir nutzen das gleiche Modell wie für unser RAG-System

# %% [markdown]
#
# ## RAGAS Datenformat
#
# RAGAS erwartet für jedes Beispiel vier Felder:
#
# | Feld | Beschreibung | Beispiel |
# |------|-------------|---------|
# | `user_input` | Die Frage | "Was ist Overfitting?" |
# | `response` | Die generierte Antwort | "Overfitting ist..." |
# | `retrieved_contexts` | Liste der Dokumente | ["Text 1", "Text 2"] |
# | `reference` | Die erwartete Antwort | "Overfitting tritt auf..." |

# %% [markdown]
#
# ## RAG-System aufbauen
#
# Wir erstellen unser RAG-System wie gewohnt:

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
    collection_name="ml_docs_ragas",
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
# ## Daten für RAGAS sammeln
#
# Wir führen unser RAG-System auf Test-Fragen aus und sammeln alle Daten:

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
# ## Zusammenfassung
#
# - RAGAS ist ein spezialisiertes Framework für RAG-Evaluierung
# - `llm_factory` erstellt ein RAGAS-kompatibles LLM aus einem `AsyncOpenAI`-Client
# - Vier Datenfelder pro Beispiel: `user_input`, `response`, `retrieved_contexts`, `reference`
# - Daten werden als Liste von Dictionaries gesammelt
#
# **Nächster Schritt**: Die RAGAS-Metriken kennenlernen!
