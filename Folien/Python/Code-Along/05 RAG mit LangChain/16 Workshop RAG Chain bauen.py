# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: RAG Chain bauen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
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
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
# Erstellen Sie Ihre Dokumente / Create your documents
my_docs = [
    # TODO: Mindestens 5 Dokumente zu einem Thema
    # TODO: At least 5 documents on a topic
]

# TODO: Create Document objects, vectorstore, and retriever
# TODO: Test with queries

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
# TODO: Create an improved system prompt
# TODO: Build RAG chain
# TODO: Test with 3 questions

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
# TODO: Create retrievers with k=1, k=2, k=3
# TODO: Build RAG chains for each
# TODO: Compare answers for the same question
