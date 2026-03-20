# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>RAG mit LangChain implementieren</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Was wir bis jetzt haben
#
# - ✅ LLMs nutzen mit LangChain
# - ✅ Text bereinigen und in Chunks aufteilen
# - ✅ Embeddings erstellen (Kosinus-Ähnlichkeit)
# - ✅ Qdrant für Vektor-Speicherung + hybride Suche
#
# **Jetzt**: Alles zusammenbauen zu einem RAG-System!

# %%
# # !pip install --root-user-action=ignore --quiet qdrant-client langchain-qdrant langfuse

# %%
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# %%
load_dotenv()

# %%
from langfuse.langchain import CallbackHandler

# %%
langfuse_handler = CallbackHandler()

# %% [markdown]
#
# ## RAG-Pipeline: Setup (einmalig)
#
# 1. Dokumente laden und chunken
# 2. Vektor-Store erstellen (mit hybrider Suche)

# %% [markdown]
#
# ## RAG-Pipeline: Chain bauen und nutzen
#
# ### Chain aufbauen (einmalig)
# 3. Retriever erstellen
# 4. Dokumente formatieren
# 5. Retrieval-Chain bauen
# 6. Prompt mit Kontext erstellen
# 7. Chat-Modell erstellen
# 8. Vollständige RAG Chain zusammenbauen
#
# ### Anfrage (bei jeder Frage)
# 9. RAG-System nutzen — **R**etrieval, **A**ugmentation, **G**eneration

# %% [markdown]
#
# ## Schritt 1: Dokumente laden und chunken

# %%

# %%
raw_documents = loader.load()

# %%
for doc in raw_documents:
    source = doc.metadata.get("source", "<unknown>")
    print(f"{source + ':':<28} {len(doc.page_content)} characters")

# %%

# %%
documents = text_splitter.split_documents(raw_documents)

# %% [markdown]
# Anzahl Chunks:

# %%
len(documents)

# %% [markdown]
#
# ## Schritt 2: Vektor-Store erstellen
#
# - Wir nutzen Qdrant für die Vektor-Speicherung

# %%
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"

# %%

# %%
API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# %%

# %%

# %% [markdown]
#
# ## Schritt 3: Retriever erstellen
#
# - Der **Retriever** durchsucht den Vektor-Store
# - Gibt die `k` ähnlichsten Dokumente zurück
# - Wird aus dem Vektor-Store erstellt

# %%

# %% [markdown]
#
# ## Retriever testen

# %%
docs_found = retriever.invoke("What is overfitting?")

# %% [markdown]
# Gefundene Chunks:

# %%
for i, doc in enumerate(docs_found, 1):
    doc_source = doc.metadata.get("source", "<unknown>")
    print(f"\n{i}. {doc_source}\n   {doc.page_content[:150]}...")


# %% [markdown]
#
# ## Schritt 4: Dokumente formatieren
#
# - Der Retriever gibt **Document-Objekte** zurück
# - Das LLM braucht aber **normalen Text**
# - `format_docs` verbindet alle Dokumente zu einem Text

# %%
def format_docs(docs, prefix="=====\n"):
    return prefix + f"\n\n{prefix}".join(doc.page_content for doc in docs)


# %%
example_docs = retriever.invoke("What is overfitting?")

# %%
print(format_docs(example_docs))

# %% [markdown]
#
# ## Schritt 5: Retrieval-Chain bauen
#
# Mit LCEL (LangChain Expression Language) können wir Schritte verketten:

# %%
retrieval_chain = retriever | format_docs

# %%

# %%

# %% [markdown]
#
# ## Schritt 6: Prompt mit Kontext
#
# Wir erstellen einen Prompt, der den Kontext und die Frage enthält:

# %%
system_prompt = (
    "Du bist ein hilfreicher Assistent. Nutze den folgenden Kontext, "
    "um die Frage zu beantworten. "
    "Wenn der Kontext keine relevanten Informationen enthält, sage: "
    "'Diese Information ist nicht in meinen Dokumenten enthalten.' "
    "Halte die Antwort kurz und prägnant.\n\n"
    "Kontext:\n\n{context}"
)

# %%
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# %% [markdown]
#
# ## Schritt 7: Chat-Modell erstellen

# %%
MINISTRAL = "mistralai/ministral-14b-2512"

# %%

# %% [markdown]
#
# ## Schritt 8: Die vollständige RAG Chain
#
# <img src="img/rag-chain-de.png" alt="RAG Chain"
#      style="display: block; margin: auto; width: 35%;"/>

# %% [markdown]
#
# ## Die Chain Schritt für Schritt
#
# 1. Die Frage geht an **zwei** Stellen gleichzeitig:
#    - `retrieval_chain` → findet relevante Dokumente → Text
#    - `RunnablePassthrough()` → gibt die Frage unverändert weiter
# 2. Beides wird in den **Prompt** eingesetzt (`{context}` und `{input}`)
# 3. Der Prompt geht ans **LLM**
# 4. `StrOutputParser()` extrahiert den Text aus der LLM-Antwort

# %%

# %% [markdown]
#
# ## Schritt 9: RAG-System nutzen

# %%
question = "What is overfitting?"

# %%

# %%

# %% [markdown]
#
# ## Quellen anzeigen
#
# Die Quellen bekommen wir separat vom Retriever:

# %%
sources = retriever.invoke(question)

# %% [markdown]
# Verwendete Quellen:

# %%
for i, doc in enumerate(sources, 1):
    print(f"{i}. {doc.page_content}...")

# %% [markdown]
#
# ## Weitere Fragen testen

# %%
questions = [
    "How do neural networks work?",
    "What is RAG?",
    "Explain gradient descent",
]

# %%
for q in questions:
    answer = rag_chain.invoke(q)
    print(f"\nQ: {q}")
    print(f"A: {answer}")

# %% [markdown]
#
# ## Was passiert bei irrelevanten Fragen?
#
# - Erinnerung: Qdrant liefert **immer** Ergebnisse
# - Unser Prompt sagt dem LLM: "Sage es, wenn der Kontext keine Antwort hat"

# %%
irrelevant = "What is the best recipe for chocolate cake?"

# %%
answer_irrelevant = rag_chain.invoke(irrelevant)

# %%
print(answer_irrelevant)

# %% [markdown]
#
# ## Prompt Engineering für RAG
#
# - Der System-Prompt ist **entscheidend** für RAG-Qualität
# - Wichtige Anweisungen im Prompt:
#   - "Nutze nur den gegebenen Kontext"
#   - "Sage, wenn die Information nicht vorhanden ist"
#   - "Halte die Antwort kurz und prägnant"
# - Ohne diese Anweisungen → LLM halluziniert trotz RAG!

# %% [markdown]
#
# ## Retrieval-Parameter: `k`
#
# - **k**: Wie viele Dokumente abrufen?
#   - Mehr → mehr Kontext, aber auch mehr Rauschen
#   - Weniger → fokussierter, aber evtl. Information fehlt

# %% [markdown]
#
# ## Retrieval-Parameter: `search_type`
#
# **search_type**:
# - `"similarity"` (Standard): Die k ähnlichsten Dokumente
# - `"similarity_score_threshold"`: Alle Dokumente über einem
#   Ähnlichkeits-Schwellenwert
#   - `score_threshold` steuert die Strenge der Suche
# - `"mmr"` (Maximal Marginal Relevance): Bevorzugt **diverse** Ergebnisse —
#   nützlich, wenn die ähnlichsten Dokumente sich zu sehr überlappen

# %%
retriever_k5 = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20})

# %%
docs_k5 = retriever_k5.invoke(question)

# %%
for i, doc in enumerate(docs_k5, 1):
    print(f"  {i}. {doc.page_content[:200]}...")

# %%
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.6, "k": 10}
)

# %%
docs_threshold = retriever_threshold.invoke(question)

# %%
for i, doc in enumerate(docs_threshold, 1):
    print(f"  {i}. {doc.page_content[:200]}...")

# %%
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    test_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": threshold, "k": 10}
    )
    docs = test_retriever.invoke(question)
    print(f"Threshold: {threshold:.2f}, number of docs: {len(docs)}")

# %% [markdown]
#
# ## Zusammenfassung
#
# - **RAG mit LangChain**: Wenige Zeilen Code für ein vollständiges System
# - **Pipeline**: Dokumente -> Chunks -> Qdrant (Hybride Suche) -> Retriever -> LLM
# - **LCEL**: Verkettung mit `|` Operator, parallele Ketten mit Dictionaries
# - **Prompt Engineering**: "Sage, wenn du es nicht weißt" verhindert Halluzinationen
# - **Retrieval-Parameter**: `k` und `search_type` steuern Kontext-Qualität
#
# **Jetzt sind Sie bereit, Ihr eigenes RAG-System zu bauen!**
