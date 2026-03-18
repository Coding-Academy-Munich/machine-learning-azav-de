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
# - ✅ Qdrant für Vektor-Speicherung + **Hybride Suche**
#
# **Jetzt**: Alles zusammenbauen zu einem RAG-System!

# %%
# !pip install --root-user-action=ignore --quiet qdrant-client langchain-qdrant

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

# %% [markdown]
#
# ## RAG-Pipeline: Setup (einmalig)
#
# 1. Dokumente laden
# 2. Text bereinigen und in Chunks aufteilen
# 3. In Qdrant speichern (mit hybrider Suche)

# %% [markdown]
#
# ## RAG-Pipeline: Anfrage (bei jeder Frage)
#
# 4. Anfrage embedden
# 5. Ähnlichste Chunks finden (**Retrieval**)
# 6. Als Kontext an LLM geben (**Augmentation**)
# 7. Antwort generieren (**Generation**)

# %% [markdown]
#
# ## Schritt 1: Dokumente laden und chunken

# %%
loader = DirectoryLoader(
    "docs/", glob="*.txt", loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

# %%
raw_documents = loader.load()

# %%
for doc in raw_documents:
    source = doc.metadata.get("source", "unknown")
    print(f"{source}: {len(doc.page_content)} characters")

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# %%
documents = text_splitter.split_documents(raw_documents)

# %% [markdown]
# Anzahl Chunks:

# %%
len(documents)

# %% [markdown]
#
# ## Schritt 2: Vektor-Store erstellen

# %%
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# %% [markdown]
#
# Erstellen Sie die Embeddings und den Qdrant Vektor-Store aus den Dokumenten.
# Verwenden Sie `RetrievalMode.HYBRID` mit `sparse_embeddings` für hybride Suche.

# %%

# %%

# %% [markdown]
#
# ## Schritt 3: Retriever erstellen
#
# - Der **Retriever** durchsucht den Vektor-Store
# - Gibt die `k` ähnlichsten Dokumente zurück
# - Wird aus dem Vektor-Store erstellt

# %% [markdown]
#
# Erstellen Sie einen Retriever aus dem Vektor-Store mit `as_retriever()`.

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
    print(f"\n{i}. {doc.page_content[:150]}...")

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
example_docs = retriever.invoke("What is overfitting?")

# %% [markdown]
# Vor `format_docs`:

# %%

# %% [markdown]
# Nach `format_docs`:

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
retrieval_chain = retriever | format_docs
context_text = retrieval_chain.invoke("What is overfitting?")

# %% [markdown]
# Kontext-Text:

# %%
