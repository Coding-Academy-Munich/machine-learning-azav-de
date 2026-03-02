# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Manuelle Evaluierung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Schritt 1: Test-Dataset erstellen
#
# - Ein gutes Test-Dataset ist die Grundlage jeder Evaluierung
# - Enthält Fragen mit erwarteten Antworten
# - Sollte verschiedene Schwierigkeitsgrade abdecken
# - Auch Fragen enthalten, die das System **nicht** beantworten kann

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
    collection_name="ml_docs_manual_eval",
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
# ## Unser Test-Dataset
#
# Fünf Fragen mit verschiedenen Schwierigkeitsgraden:

# %%
test_dataset = [
    {
        "question": "Was ist Overfitting?",
        "expected_answer": "Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt und nicht auf neue Daten generalisiert.",
    },
    {
        "question": "Wie werden Neural Networks trainiert?",
        "expected_answer": "Neural Networks werden durch Backpropagation und Gradient Descent trainiert.",
    },
    {
        "question": "Was ist der Vorteil von Random Forests?",
        "expected_answer": "Random Forests kombinieren viele Decision Trees und sind oft genauer als einzelne Modelle.",
    },
    {
        "question": "Wie hilft RAG bei LLMs?",
        "expected_answer": "RAG hilft LLMs, präziser mit spezifischem Wissen zu antworten.",
    },
    {
        "question": "Was ist der Unterschied zwischen Bagging und Boosting?",
        "expected_answer": "Diese Information ist nicht in den Dokumenten enthalten.",
    },
]

# %% [markdown]
#
# ## Schritt 2: RAG-Pipeline ausführen
#
# Für jede Frage sammeln wir:
# - Die Frage selbst
# - Die abgerufenen Dokumente
# - Die generierte Antwort

# %%
results = []
for item in test_dataset:
    question = item["question"]
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    results.append(
        {
            "question": question,
            "expected_answer": item["expected_answer"],
            "retrieved_contexts": [doc.page_content for doc in docs],
            "generated_answer": answer,
        }
    )

# %%

# %% [markdown]
#
# ## Schritt 3: Ergebnisse manuell bewerten
#
# Schauen wir uns jede Frage im Detail an:

# %%
for i, r in enumerate(results, 1):
    print(f"{'='*60}")
    print(f"Frage/Question {i}: {r['question']}")
    print(f"\nErwartete Antwort / Expected:")
    print(f"  {r['expected_answer']}")
    print(f"\nGenerierte Antwort / Generated:")
    print(f"  {r['generated_answer'][:200]}")
    print(f"\nAbgerufene Dokumente / Retrieved docs:")
    for j, ctx in enumerate(r["retrieved_contexts"], 1):
        print(f"  {j}. {ctx[:80]}...")
    print()

# %% [markdown]
#
# ## Schritt 4: Bewertung
#
# Für jede Antwort prüfen wir manuell:
#
# | Kriterium | Frage |
# |-----------|-------|
# | **Retrieval** | Wurden relevante Dokumente gefunden? |
# | **Faithfulness** | Basiert die Antwort auf den Dokumenten? |
# | **Relevancy** | Beantwortet sie die Frage? |
# | **Korrektheit** | Stimmt die Antwort mit der Referenz überein? |

# %%
manual_scores = []
for i, r in enumerate(results, 1):
    print(f"Frage/Question {i}: {r['question']}")
    print(f"  Antwort/Answer: {r['generated_answer'][:100]}...")
    print()
    manual_scores.append(
        {
            "question": r["question"],
            "retrieval_ok": True,
            "faithful": True,
            "relevant": True,
        }
    )

# %% [markdown]
#
# - In der Praxis würden wir hier jede Antwort einzeln bewerten
# - Für 5 Fragen ist das machbar...

# %% [markdown]
#
# ## Das Problem der manuellen Evaluierung
#
# - 5 Fragen → **~10 Minuten** Bewertungszeit
# - 50 Fragen → **~2 Stunden**
# - 500 Fragen → **~1 ganzer Arbeitstag**
#
# Und nach jeder Änderung am System müssen wir alles wiederholen!
#
# **Wir brauchen Automatisierung.**

# %% [markdown]
#
# ## Zusammenfassung
#
# - Ein gutes Test-Dataset ist die Grundlage
# - Für jede Frage sammeln wir: Frage, Kontext, Antwort, Referenz
# - Manuelle Bewertung ist der Gold-Standard, aber nicht skalierbar
# - **Nächster Schritt**: LangSmith für automatische Evaluierung!
