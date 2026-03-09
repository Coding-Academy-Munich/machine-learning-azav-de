# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>RAG Chain mit LangChain</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


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
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

# %%
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# %%
docs_content = [
    """Linear Regression ist eine Methode des überwachten Lernens.
    Sie modelliert lineare Beziehungen zwischen Features und Zielvariable.
    Die Normalgleichung kann verwendet werden, um optimale Parameter zu finden.""",

    """Neural Networks bestehen aus Schichten von Neuronen.
    Jedes Neuron berechnet eine gewichtete Summe und wendet eine Aktivierungsfunktion an.
    Training erfolgt durch Backpropagation und Gradient Descent.""",

    """Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt.
    Regularisierung und Cross-Validation helfen, Overfitting zu vermeiden.
    Ein gutes Modell generalisiert auf neue, ungesehene Daten.""",

    """Large Language Models sind sehr große neuronale Netze.
    Sie werden auf Milliarden von Wörtern trainiert.
    RAG hilft LLMs, präziser mit spezifischem Wissen zu antworten."""
]

# %%
documents = [Document(page_content=doc) for doc in docs_content]

# %%
vectorstore = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name="ml_docs_chain",
    location=":memory:",
    retrieval_mode=RetrievalMode.HYBRID,
)

# %%
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %% [markdown]
#
# ## Schritt 4b: Prompt mit Kontext
#
# Wir erstellen einen Prompt, der den Kontext und die Frage enthält:

# %%
system_prompt = (
    "Du bist ein hilfreicher Assistent. Nutze den folgenden Kontext, "
    "um die Frage zu beantworten. "
    "Wenn der Kontext keine relevanten Informationen enthält, sage: "
    "'Diese Information ist nicht in meinen Dokumenten enthalten.' "
    "Halte die Antwort präzise.\n\n"
    "Kontext: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
    temperature=0
)

# %% [markdown]
#
# ## Schritt 4c: Die vollständige RAG Chain
#
# ```
# Frage ──→ retriever | format_docs ──→ context
#       └──→ RunnablePassthrough()  ──→ input
#                                       ↓
#                                    prompt
#                                       ↓
#                                      llm
#                                       ↓
#                                 StrOutputParser()
#                                       ↓
#                                    Antwort
# ```

# %% [markdown]
#
# ## Die Chain Schritt für Schritt
#
# 1. Die Frage geht an **zwei** Stellen gleichzeitig:
#    - `retriever | format_docs` → findet relevante Dokumente → Text
#    - `RunnablePassthrough()` → gibt die Frage unverändert weiter
# 2. Beides wird in den **Prompt** eingesetzt (`{context}` und `{input}`)
# 3. Der Prompt geht ans **LLM**
# 4. `StrOutputParser()` extrahiert den Text aus der LLM-Antwort

# %% [markdown]
#
# Bauen Sie die RAG Chain mit LCEL. Verwenden Sie ein Dictionary mit `"context"` und
# `"input"` als Schlüssel, dann verketten Sie mit `prompt`, `llm` und `StrOutputParser()`.

# %%

# %%

# %% [markdown]
#
# ## Schritt 5: RAG-System nutzen

# %%
question = "Was ist Overfitting?"

# %%

# %% [markdown]
# Frage:

# %%

# %% [markdown]
# Antwort:

# %%

# %% [markdown]
#
# ## Quellen anzeigen
#
# Die Quellen bekommen wir separat vom Retriever:

# %%

# %% [markdown]
#
# ## Weitere Fragen testen

# %%

# %% [markdown]
#
# ## Was passiert bei irrelevanten Fragen?
#
# - Erinnerung: Qdrant liefert **immer** Ergebnisse
# - Unser Prompt sagt dem LLM: "Sage es, wenn der Kontext keine Antwort hat"

# %%

# %% [markdown]
# Frage:

# %%

# %% [markdown]
# Antwort:

# %%

# %% [markdown]
#
# ## Prompt Engineering für RAG
#
# - Der System-Prompt ist **entscheidend** für RAG-Qualität
# - Wichtige Anweisungen im Prompt:
#   - "Nutze nur den gegebenen Kontext"
#   - "Sage, wenn die Information nicht vorhanden ist"
#   - "Halte die Antwort präzise"
# - Ohne diese Anweisungen → LLM halluziniert trotz RAG!

# %% [markdown]
#
# ## Retrieval-Parameter: `k`
#
# - **k**: Wie viele Dokumente abrufen? (Standard: 4)
#   - Mehr → mehr Kontext, aber auch mehr Rauschen
#   - Weniger → fokussierter, aber evtl. Information fehlt

# %% [markdown]
#
# ## Retrieval-Parameter: `search_type`
#
# - **search_type**:
#   - `"similarity"` (Standard): Die k ähnlichsten Dokumente
#   - `"mmr"` (Maximal Marginal Relevance): Bevorzugt **diverse** Ergebnisse —
#     nützlich, wenn die ähnlichsten Dokumente sich zu sehr überlappen

# %%
retriever_k3 = vectorstore.as_retriever(search_kwargs={"k": 3})

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **RAG mit LangChain**: Wenige Zeilen Code für ein vollständiges System
# - **Pipeline**: Dokumente -> Chunks -> Qdrant (Hybrid Search) -> Retriever -> LLM
# - **LCEL**: Verkettung mit `|` Operator, z.B.
#   `{"context": retriever | format_docs, "input": RunnablePassthrough()}`
# - **Prompt Engineering**: "Sage, wenn du es nicht weißt" verhindert Halluzinationen
# - **Retrieval-Parameter**: `k` und `search_type` steuern Kontext-Qualität
#
# **Jetzt sind Sie bereit, Ihr eigenes RAG-System zu bauen!**
