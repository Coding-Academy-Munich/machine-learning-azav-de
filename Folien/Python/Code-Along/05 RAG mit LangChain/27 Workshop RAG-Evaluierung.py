# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop: RAG-Evaluierung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Workshop-Ziel
#
# In diesem Workshop werden Sie:
# 1. Ein RAG-System aufbauen und manuell testen
# 2. Mit LangSmith automatisch evaluieren
# 3. Mit RAGAS standardisierte Metriken berechnen
# 4. Das System verbessern und den Effekt messen
#
# **Sie arbeiten mit dem gleichen RAG-System wie im vorherigen Workshop!**

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
from openai import AsyncOpenAI
from langsmith import Client
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall
from typing import TypedDict
import pandas as pd

# %%
load_dotenv()

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
    temperature=0,
)

client = Client()

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
evaluator_llm = llm_factory(
    "mistralai/ministral-14b-2512",
    provider="openai",
    client=openai_client,
)
evaluator_embeddings = embedding_factory(
    "openai",
    model="openai/text-embedding-3-small",
    client=openai_client,
)

faithfulness = Faithfulness(llm=evaluator_llm)
answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
context_recall = ContextRecall(llm=evaluator_llm)

# %%
sample_docs = [
    """# Machine Learning Kurs - Kapitel 1: Einführung

Machine Learning ist ein Teilgebiet der Künstlichen Intelligenz.
Computer lernen aus Daten, ohne explizit programmiert zu werden.

## Arten von Machine Learning
- Überwachtes Lernen: Lernen mit gelabelten Daten
- Unüberwachtes Lernen: Muster in ungelabelten Daten finden
- Reinforcement Learning: Lernen durch Belohnung und Bestrafung
""",
    """# Machine Learning Kurs - Kapitel 2: Linear Regression

Linear Regression ist eine grundlegende Methode des überwachten Lernens.

## Konzept
- Modelliert lineare Beziehungen zwischen Features (X) und Ziel (y)
- Formel: y = w*X + b
- w sind die Gewichte, b ist der Bias

## Training
- Normalgleichung für exakte Lösung
- Gradient Descent für große Datensätze
- Loss-Funktion: Mean Squared Error (MSE)
""",
    """# Machine Learning Kurs - Kapitel 3: Neural Networks

Neural Networks sind inspiriert von biologischen Neuronen.

## Aufbau
- Input Layer, Hidden Layers, Output Layer
- Jedes Neuron: gewichtete Summe + Aktivierungsfunktion
- Typische Aktivierungsfunktionen: ReLU, Sigmoid, Tanh

## Training
- Backpropagation berechnet Gradienten
- Gradient Descent passt Gewichte an
- Lernrate bestimmt Schrittgröße
""",
    """# Machine Learning Kurs - Kapitel 4: Overfitting

Overfitting ist ein häufiges Problem beim Machine Learning.

## Symptome
- Sehr gute Performance auf Trainingsdaten
- Schlechte Performance auf Testdaten (neue Daten)

## Lösungen
- Mehr Trainingsdaten sammeln
- Regularisierung (L1, L2)
- Cross-Validation zur Modellauswahl
- Early Stopping beim Training
- Dropout bei Neural Networks
""",
    """# Machine Learning Kurs - Kapitel 5: Decision Trees und Random Forests

Decision Trees teilen Daten anhand von Entscheidungsregeln auf.

## Decision Trees
- Einfach zu interpretieren
- Können overfitting (zu tiefe Bäume)
- Verwenden Information Gain oder Gini Impurity

## Random Forests
- Kombination vieler Decision Trees (Ensemble-Methode)
- Jeder Baum trainiert auf zufälliger Teilmenge
- Oft genauer als einzelne Modelle
- Feature Importance zeigt wichtigste Features
""",
    """# Machine Learning Kurs - Kapitel 6: LLMs und RAG

Large Language Models (LLMs) sind sehr große neuronale Netze.

## LLMs
- Trainiert auf Milliarden von Wörtern
- Können Texte generieren, übersetzen, zusammenfassen
- Prompt Engineering ist wichtig für gute Ergebnisse

## RAG (Retrieval Augmented Generation)
- Ergänzt LLMs mit externem Wissen
- Reduziert Halluzinationen
- Dokumente werden in Vektoren umgewandelt
- Ähnlichste Dokumente werden als Kontext bereitgestellt
""",
]

# %%
documents = [Document(page_content=doc) for doc in sample_docs]

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small",
)

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

vectorstore = QdrantVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name="workshop_eval",
    location=":memory:",
    retrieval_mode=RetrievalMode.HYBRID,
)

# %%
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
    [("system", system_prompt), ("human", "{input}")]
)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %%
print(f"RAG-System bereit / ready: {len(splits)} Chunks")

# %%
test_dataset = [
    {
        "question": "Was ist Machine Learning?",
        "reference": "Machine Learning ist ein Teilgebiet der KI, bei dem Computer aus Daten lernen, ohne explizit programmiert zu werden.",
    },
    {
        "question": "Was ist Overfitting und wie kann man es vermeiden?",
        "reference": "Overfitting tritt auf, wenn ein Modell auf Trainingsdaten gut, aber auf neuen Daten schlecht funktioniert. Lösungen: mehr Daten, Regularisierung, Cross-Validation, Early Stopping, Dropout.",
    },
    {
        "question": "Wie funktioniert Linear Regression?",
        "reference": "Linear Regression modelliert lineare Beziehungen zwischen Features und Zielvariable mit der Formel y = w*X + b. Training mit Normalgleichung oder Gradient Descent.",
    },
    {
        "question": "Was ist der Unterschied zwischen Decision Trees und Random Forests?",
        "reference": "Decision Trees sind einzelne Bäume mit Entscheidungsregeln. Random Forests kombinieren viele Trees als Ensemble und sind oft genauer.",
    },
    {
        "question": "Wie hilft RAG bei LLMs?",
        "reference": "RAG ergänzt LLMs mit externem Wissen, reduziert Halluzinationen und stellt relevante Dokumente als Kontext bereit.",
    },
    {
        "question": "Was sind die drei Arten von Machine Learning?",
        "reference": "Überwachtes Lernen (gelabelte Daten), Unüberwachtes Lernen (Muster in ungelabelten Daten) und Reinforcement Learning (Belohnung/Bestrafung).",
    },
    {
        "question": "Was ist Backpropagation?",
        "reference": "Backpropagation ist der Algorithmus, der Gradienten in Neural Networks berechnet, um die Gewichte durch Gradient Descent anzupassen.",
    },
    {
        "question": "Wie funktioniert Support Vector Machine?",
        "reference": "Diese Information ist nicht in den Dokumenten enthalten.",
    },
]

# %% [markdown]
#
# ## Aufgabe 1: Manuelle Evaluierung
#
# Führen Sie das RAG-System auf allen Test-Fragen aus.
# Identifizieren Sie **zwei Fragen**, bei denen das System schlecht abschneidet.
#
# Hinweise:
# - Prüfen Sie, ob die abgerufenen Dokumente relevant sind
# - Prüfen Sie, ob die Antwort auf den Dokumenten basiert
# - Prüfen Sie, ob die Frage beantwortet wurde

# %%

# %%

# %% [markdown]
#
# ## Aufgabe 2: LangSmith Evaluierung
#
# 1. Erstellen Sie ein LangSmith Dataset mit den Test-Fragen
# 2. Schreiben Sie einen heuristischen Evaluator (Antwortlänge)
# 3. Schreiben Sie einen LLM-als-Richter Evaluator (Korrektheit)
# 4. Führen Sie `client.evaluate()` aus

# %%

# %%

# %%

# %%

# %%

# %%

# %%
print("Aufgabe 2 abgeschlossen! / Task 2 complete!")
print("Schauen Sie sich die Ergebnisse in LangSmith an / Check results in LangSmith")

# %% [markdown]
#
# ## Aufgabe 3: RAGAS Evaluierung
#
# 1. Berechnen Sie Faithfulness, AnswerRelevancy und ContextRecall mit `abatch_score()`
# 2. Zeigen Sie die Durchschnittswerte an
# 3. Zeigen Sie die Ergebnisse pro Frage an

# %%

# %%

# %%

# %%

# %%

# %%
print("Aufgabe 3 abgeschlossen! / Task 3 complete!")

# %% [markdown]
#
# ## Aufgabe 4: Verbessern und erneut evaluieren
#
# Wählen Sie **eine** Verbesserung:
# - k von 2 auf 4 erhöhen (mehr Dokumente abrufen)
# - System-Prompt verbessern (strengere Anweisungen)
# - Chunk-Größe ändern
#
# Führen Sie die RAGAS-Evaluierung erneut aus und vergleichen Sie!

# %%

# %%

# %%
print("Vergleich / Comparison:")
print(f"{'Metrik/Metric':<25} {'Baseline (k=2)':>15} {'Verbessert/Improved (k=4)':>25}")
print("-" * 65)
for label, base_res, imp_res in [("Faithfulness", faith_results, faith_improved), ("Answer Relevancy", relevancy_results, relevancy_improved), ("Context Recall", recall_results, recall_improved)]:
    base = sum(r.value for r in base_res) / len(base_res)
    imp = sum(r.value for r in imp_res) / len(imp_res)
    print(f"{label:<25} {base:>15.2f} {imp:>25.2f}")

# %% [markdown]
#
# ## Aufgabe 5 (Optional): Custom Evaluator
#
# Schreiben Sie einen Evaluator, der prüft, ob die Antwort bestimmte
# Schlüsselwörter aus dem Kontext verwendet.

# %% [markdown]
#
# ## Aufgabe 6 (Optional): Evaluierungs-Dashboard
#
# Erstellen Sie ein einfaches Gradio-Interface, das:
# - Die RAGAS-Scores anzeigt
# - Einzelne Fragen mit ihren Scores darstellt
# - Einen Vergleich vor/nach Verbesserung zeigt

# %% [markdown]
#
# ## Zusammenfassung
#
# **Was Sie gelernt haben:**
# - Manuelle Evaluierung: Gold-Standard, aber nicht skalierbar
# - LangSmith: Datasets, Evaluatoren, Experimente vergleichen
# - RAGAS: Standardisierte Metriken (Faithfulness, Relevancy, Recall)
# - Verbesserungszyklus: Messen → Verbessern → Erneut messen
#
# **Wichtigste Erkenntnis:**
# - Ohne Evaluierung ist RAG-Optimierung nur Raten
# - Mit Evaluierung können Sie datenbasiert verbessern
#
# **Nächster Schritt**: LLM Orchestration — Noch komplexere Workflows!
