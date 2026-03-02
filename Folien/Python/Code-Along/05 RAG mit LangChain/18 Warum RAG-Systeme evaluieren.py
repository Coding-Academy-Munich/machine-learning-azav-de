# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Warum RAG-Systeme evaluieren?</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Ihr RAG-System funktioniert... oder?
#
# - Sie haben ein RAG-System gebaut
# - Es beantwortet Fragen
# - Die Antworten sehen gut aus...
#
# **Aber woher wissen Sie, ob es wirklich gut funktioniert?**

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

# %%
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
    collection_name="ml_docs_eval",
    location=":memory:",
    retrieval_mode=RetrievalMode.HYBRID,
)

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
# ## Live-Demo: Einfache Fragen
#
# Fragen, die unser System gut beantwortet:

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Live-Demo: Schwierige Fragen
#
# Aber was passiert bei kniffligeren Fragen?

# %%

# %%

# %% [markdown]
#
# - Die Dokumente erwähnen Regularisierung, aber erklären den Unterschied nicht
# - Das LLM hat vielleicht trotzdem geantwortet — aber aus eigenem Wissen!
# - **Das ist eine Halluzination**: Die Antwort kommt nicht aus dem Kontext

# %%

# %%

# %% [markdown]
#
# ## Was kann schiefgehen?
#
# Vier typische Fehlerarten bei RAG-Systemen:
#
# 1. **Falsche Dokumente** abgerufen
#    - Die Suche findet irrelevante Texte
# 2. **Richtige Dokumente, falsche Antwort**
#    - Das LLM interpretiert den Kontext falsch
# 3. **Halluzination**
#    - Die Antwort enthält Informationen, die nicht im Kontext stehen
# 4. **Antwort geht am Thema vorbei**
#    - Die Frage wird nicht wirklich beantwortet

# %% [markdown]
#
# ## Warum reicht "sieht gut aus" nicht?
#
# - **Skala**: 5 Fragen können wir manuell prüfen, 500 nicht
# - **Konsistenz**: Menschen bewerten unterschiedlich
# - **Vergleichbarkeit**: Ist Version A besser als Version B?
# - **Regression**: Hat eine Änderung etwas verschlechtert?
#
# Wir brauchen **systematische Evaluierung**!

# %% [markdown]
#
# ## Drei Arten der Evaluierung
#
# | Art | Beschreibung | Aufwand |
# |-----|-------------|---------|
# | **Manuell** | Menschen bewerten Antworten | Hoch, aber Gold-Standard |
# | **Heuristisch** | Einfache Regeln (Länge, Keywords) | Niedrig, aber begrenzt |
# | **LLM-als-Richter** | Ein anderes LLM bewertet | Mittel, überraschend gut |

# %% [markdown]
#
# ## LLM-als-Richter
#
# - Ein separates LLM bewertet die Antworten
# - Vorteile:
#   - Skalierbar: Kann tausende Antworten bewerten
#   - Konsistent: Gleiche Kriterien für alle
#   - Schnell: Sekunden statt Stunden
# - **Wichtig**: Der Richter-LLM muss nicht der gleiche sein wie der RAG-LLM

# %% [markdown]
#
# ## Der Evaluierungs-Workflow
#
# ```
# Test-Fragen + Erwartete Antworten
#         |
#     RAG-Pipeline ausführen
#         |
#     Sammeln: Frage, Kontext, Antwort, Referenz
#         |
#     Mit Metriken evaluieren
#         |
#     Verbessern und erneut evaluieren
# ```
#
# Dieser Zyklus ist der Schlüssel zu einem guten RAG-System!

# %% [markdown]
#
# ## Unsere Werkzeuge
#
# Wir werden zwei Tools verwenden:
#
# - **LangSmith** (kennen Sie schon!)
#   - Für den Evaluierungs-Workflow: Datasets, Evaluatoren, Experimente
#   - Visuelle Vergleiche in der Web-Oberfläche
# - **RAGAS** (neu)
#   - Spezialisiertes Framework für RAG-Evaluierung
#   - Standardisierte Metriken: Faithfulness, Relevancy, Context Recall
#
# "LangSmith gibt uns den Workflow, RAGAS gibt uns die Metriken"

# %% [markdown]
#
# ## Zusammenfassung
#
# - RAG-Systeme können auf vier Arten versagen
# - "Sieht gut aus" reicht nicht für professionelle Systeme
# - Systematische Evaluierung ermöglicht:
#   - Vergleich verschiedener Konfigurationen
#   - Erkennung von Regressionen
#   - Datenbasierte Verbesserungen
# - Drei Ansätze: manuell, heuristisch, LLM-als-Richter
# - Unsere Tools: **LangSmith** + **RAGAS**
