# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>RAG mit LangChain implementieren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bis jetzt haben
#
# - ✅ LLMs nutzen mit LangChain
# - ✅ Text bereinigen und in Chunks aufteilen
# - ✅ Embeddings erstellen (Kosinus-Ähnlichkeit)
# - ✅ Qdrant für Vektor-Speicherung + **Hybrid Search**
#
# **Jetzt**: Alles zusammenbauen zu einem RAG-System!

# %%
# ! pip install qdrant-client langchain-qdrant

# %%
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# %%
load_dotenv()

# %% [markdown]
#
# ## RAG-Pipeline Übersicht
#
# **Einmalig (Setup)**:
# 1. Dokumente laden
# 2. Text bereinigen und in Chunks aufteilen
# 3. In Qdrant speichern (mit Hybrid Search)
#
# **Bei jeder Anfrage**:
# 4. Anfrage embedden
# 5. Ähnlichste Chunks finden (Retrieval)
# 6. Als Kontext an LLM geben (Augmentation)
# 7. Antwort generieren (Generation)

# %% [markdown]
#
# ## Schritt 1: Dokumente laden und chunken

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

# %% [markdown]
#
# Für längere Dokumente: Chunks erstellen

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# %%

# %% [markdown]
#
# ## Schritt 2: Vektor-Store erstellen

# %%
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")


# %%
# TODO: Create embeddings and Qdrant vector store from documents
# Use RetrievalMode.HYBRID with sparse_embeddings for hybrid search

# %%

# %% [markdown]
#
# ## Schritt 3: Retriever erstellen
#
# - Der **Retriever** durchsucht den Vektor-Store
# - Gibt die `k` ähnlichsten Dokumente zurück
# - Wird aus dem Vektor-Store erstellt

# %%
# TODO: Create retriever from vectorstore
# retriever = vectorstore.as_retriever(...)

# %% [markdown]
#
# ## Retriever testen

# %%

# %%

# %% [markdown]
#
# ## Schritt 4: RAG Chain — Warum format_docs?
#
# - Der Retriever gibt **Document-Objekte** zurück
# - Das LLM braucht aber **normalen Text**
# - `format_docs` verbindet alle Dokumente zu einem Text

# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %%

# %%

# %% [markdown]
#
# ## Schritt 4a: Retrieval + Formatierung
#
# Mit LCEL (LangChain Expression Language) können wir Schritte verketten:
#
# ```
# retriever | format_docs
# ```
#
# - `|` bedeutet: "Output von links wird Input von rechts"
# - Frage → Retriever → Dokumente → format_docs → Text

# %%

# %%

# %% [markdown]
#
# ## Schritt 4b: Prompt mit Kontext
#
# Wir erstellen einen Prompt, der den Kontext und die Frage enthält:

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

# %%
# TODO: Build the RAG chain using LCEL
# rag_chain = (
#     {"context": ..., "input": ...}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# %%

# %% [markdown]
#
# ## Schritt 5: RAG-System nutzen

# %%
question = "Was ist Overfitting?"

# %%

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
# ## Retrieval-Parameter anpassen
#
# - **k**: Wie viele Dokumente abrufen? (Standard: 4)
#   - Mehr → mehr Kontext, aber auch mehr Rauschen
#   - Weniger → fokussierter, aber evtl. Information fehlt
# - **search_type**: `"similarity"` (Standard) oder `"mmr"` (diverse Ergebnisse)

# %%
retriever_k3 = vectorstore.as_retriever(search_kwargs={"k": 3})

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **RAG mit LangChain**: Wenige Zeilen Code für ein vollständiges System
# - **Pipeline**: Dokumente → Chunks → Qdrant (Hybrid Search) → Retriever → LLM
# - **LCEL**: Verkettung mit `|` Operator
#   - `{"context": retriever | format_docs, "input": RunnablePassthrough()}`
# - **Prompt Engineering**: Entscheidend für gute RAG-Antworten
#   - "Sage, wenn du es nicht weißt" verhindert Halluzinationen
# - **Quellen**: Separat abrufbar über `retriever.invoke()`
#
# **Jetzt sind Sie bereit, Ihr eigenes RAG-System zu bauen!**
