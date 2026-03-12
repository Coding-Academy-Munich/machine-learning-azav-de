# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: RAG Chain bauen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Workshop: RAG Chain bauen
#
# In diesem Workshop üben Sie:
# 1. Eigene Dokumente zum RAG-System hinzufügen
# 2. Den System-Prompt für bessere Antworten optimieren
# 3. Retrieval-Parameter anpassen und vergleichen

# %%
# !pip install qdrant-client langchain-qdrant

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

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
    temperature=0
)

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %% [markdown]
#
# ## Aufgabe 1: Eigene Dokumente hinzufügen
#
# Erstellen Sie einen Vektor-Store mit mindestens 5 Dokumenten zu einem Thema
# Ihrer Wahl (z.B. Kochen, Sport, Geschichte, Musik...).
# Testen Sie den Retriever mit 2–3 Anfragen.

# %%

# %%

# %%

# %%
assert len(my_docs) >= 5

# %%
docs_found = my_retriever.invoke("test")

# %%
assert len(docs_found) == 2

# %%

# %% [markdown]
#
# ## Aufgabe 2: System-Prompt verbessern
#
# Erstellen Sie eine RAG Chain und verbessern Sie den System-Prompt:
# - Die Antwort soll in **Stichpunkten** gegeben werden
# - Das LLM soll die **verwendete Quelle zitieren**
# - Bei irrelevanten Fragen soll es sagen: "Dazu habe ich keine Informationen"
#
# Testen Sie mit 3 Fragen (2 relevante, 1 irrelevante).

# %%

# %%

# %%
test_answer = my_rag_chain.invoke("test")

# %%
assert isinstance(test_answer, str)
assert len(test_answer) > 0

# %%
test_questions = [
    "Wie macht man Risotto?",
    "Was ist Tiramisu?",
    "Wie repariere ich mein Fahrrad?",
]

# %%

# %% [markdown]
#
# ## Aufgabe 3: Retrieval-Parameter vergleichen
#
# Testen Sie verschiedene Werte für `k` (1, 2, 3) und vergleichen Sie die
# Antwortqualität. Verwenden Sie dieselbe Frage für jeden Wert.
#
# Beobachten Sie: Wie verändert sich die Antwort mit mehr/weniger Kontext?

# %%
question = "Welche italienischen Gerichte gibt es?"

# %%
