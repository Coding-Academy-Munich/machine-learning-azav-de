# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LangSmith Datasets</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Von Tracing zu Evaluierung
#
# Sie kennen LangSmith bereits für **Tracing**:
# - Jeden LLM-Aufruf sehen
# - Inputs, Outputs, Latenz, Token-Verbrauch
#
# Jetzt nutzen wir LangSmith für **Evaluierung**:
# - **Datasets**: Sammlungen von Test-Beispielen
# - **Evaluatoren**: Funktionen, die Outputs bewerten
# - **Experimente**: Systematische Vergleiche

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

# %%
load_dotenv()

# %% [markdown]
#
# ## LangSmith Client
#
# - Der `Client` verbindet uns mit der LangSmith API
# - Nutzt automatisch die Umgebungsvariablen aus `.env`

# %%
client = Client()

# %% [markdown]
#
# ## Was ist ein Dataset?
#
# Ein Dataset in LangSmith enthält:
# - **Inputs**: Die Eingaben für unser System (z.B. Fragen)
# - **Reference Outputs**: Die erwarteten Ergebnisse
#
# ```
# Dataset: "RAG Evaluation - ML Kurs"
# ┌─────────────────────────────────┬──────────────────────────────┐
# │ Input                           │ Reference Output             │
# ├─────────────────────────────────┼──────────────────────────────┤
# │ "Was ist Overfitting?"          │ "Modell lernt Daten aus..."  │
# │ "Wie funktionieren NNs?"        │ "Backpropagation und..."     │
# └─────────────────────────────────┴──────────────────────────────┘
# ```

# %% [markdown]
#
# ## Dataset erstellen

# %%
dataset_name = "RAG Eval - ML Kurs"

# %%
existing = list(client.list_datasets(dataset_name=dataset_name))

# %%

# %%
if existing:
    client.delete_dataset(dataset_id=existing[0].id)

# %%
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Test-Fragen für RAG-Evaluierung des ML-Kurses",
)

# %%

# %% [markdown]
#
# ## Test-Beispiele hinzufügen
#
# Jedes Beispiel hat:
# - `inputs`: Was unser RAG-System als Eingabe bekommt
# - `outputs`: Die erwartete/richtige Antwort

# %%
examples = [
    {
        "inputs": {"question": "Was ist Overfitting?"},
        "outputs": {
            "answer": "Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt und nicht auf neue Daten generalisiert."
        },
    },
    {
        "inputs": {"question": "Wie werden Neural Networks trainiert?"},
        "outputs": {
            "answer": "Neural Networks werden durch Backpropagation und Gradient Descent trainiert."
        },
    },
    {
        "inputs": {"question": "Was ist der Vorteil von Random Forests?"},
        "outputs": {
            "answer": "Random Forests kombinieren viele Decision Trees und sind oft genauer als einzelne Modelle."
        },
    },
    {
        "inputs": {"question": "Wie hilft RAG bei LLMs?"},
        "outputs": {
            "answer": "RAG hilft LLMs, präziser mit spezifischem Wissen zu antworten, indem relevante Dokumente als Kontext bereitgestellt werden."
        },
    },
    {
        "inputs": {"question": "Was ist Linear Regression?"},
        "outputs": {
            "answer": "Linear Regression ist eine Methode des überwachten Lernens, die lineare Beziehungen zwischen Features und Zielvariable modelliert."
        },
    },
    {
        "inputs": {"question": "Was ist der Unterschied zwischen Bagging und Boosting?"},
        "outputs": {
            "answer": "Diese Information ist nicht in den Dokumenten enthalten."
        },
    },
]

# %%
create_examples_result = client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id,
)

# %%

# %%

# %% [markdown]
#
# ## Dataset in LangSmith ansehen
#
# Öffnen Sie https://smith.langchain.com:
#
# 1. Navigieren Sie zu **Datasets**
# 2. Finden Sie "RAG Eval - ML Kurs"
# 3. Sehen Sie sich die Beispiele an
#
# Das Dataset bleibt gespeichert und kann immer wieder verwendet werden!

# %% [markdown]
#
# ## Dataset überprüfen

# %%
for example in client.list_examples(dataset_id=dataset.id):
    q = example.inputs["question"]
    a = example.outputs["answer"][:60]
    print(f"Q: {q}")
    print(f"A: {a}...")
    print()

# %% [markdown]
#
# ## RAG-System aufbauen
#
# Wir erstellen das gleiche RAG-System wie bisher:

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
    collection_name="ml_docs_ls_eval",
    location=":memory:",
    retrieval_mode=RetrievalMode.HYBRID,
)

# %%
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


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
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %% [markdown]
#
# ## Zusammenfassung
#
# - LangSmith **Datasets** speichern Test-Beispiele mit erwarteten Antworten
# - Datasets sind **persistent** — einmal erstellen, immer wieder verwenden
# - Jedes Beispiel hat **inputs** und **outputs**
# - Das Dataset ist die Grundlage für automatische Evaluierung
#
# **Nächster Schritt**: Evaluatoren definieren!
