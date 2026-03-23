# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LangGraph: Graphen für LLM-Workflows</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Von Chains zu Graphs
#
# - **Chains**: Linear (A → B → C)
# - **Graphs**: Flexibel (Verzweigungen, Schleifen, Bedingungen)
# - **LangGraph**: Bibliothek für Graph-basierte Workflows
#
# LangGraph ist der **moderne Standard** für komplexe LLM-Anwendungen!

# %%
# !pip install langgraph

# %%
import os
from dotenv import load_dotenv

load_dotenv()

# %% [markdown]
#
# ## Wann Graphs statt Chains?
#
# - **Bedingte Logik**: Verschiedene Pfade je nach Input
# - **Schleifen**: Wiederhole Schritte bis Bedingung erfüllt
# - **Agentic Workflows**: LLM entscheidet nächsten Schritt
# - **Komplexe Workflows**: Mehrere parallele Prozesse
# - **Tool Calling**: LLM ruft externe Tools auf

# %% [markdown]
#
# ## LangGraph Grundkonzepte
#
# 1. **State**: Gemeinsamer Zustand, der durch den Graph fließt
# 2. **Nodes**: Funktionen, die den State verarbeiten
# 3. **Edges**: Verbindungen zwischen Nodes
# 4. **Conditional Edges**: Verzweigungen basierend auf State

# %%
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# %% [markdown]
#
# ## State definieren mit TypedDict
#
# Der State ist ein Dictionary mit definierten Feldern:

# %%
class SimpleState(TypedDict):
    """State for our simple graph."""
    message: str
    processed: bool

# %% [markdown]
#
# ## Erster Graph: Hello World
#
# Ein minimaler Graph mit zwei Nodes:

# %%
def greet(state: SimpleState) -> SimpleState:
    """Add greeting to message."""
    return {"message": f"Hello! {state['message']}", "processed": False}

def process(state: SimpleState) -> SimpleState:
    """Mark as processed."""
    return {"message": state["message"].upper(), "processed": True}

# %%
# TODO: Create a StateGraph with SimpleState

# %%

# %% [markdown]
#
# ## Conditional Edges: Verzweigungen
#
# Manchmal brauchen wir verschiedene Pfade basierend auf dem State:

# %%
class RouterState(TypedDict):
    """State with routing capability."""
    text: str
    sentiment: str
    response: str

# %%
def analyze_sentiment(state: RouterState) -> RouterState:
    """Simple sentiment analysis (mock)."""
    text = state["text"].lower()
    if any(word in text for word in ["happy", "great", "love", "good"]):
        return {"sentiment": "positive", "text": state["text"], "response": ""}
    elif any(word in text for word in ["sad", "bad", "hate", "terrible"]):
        return {"sentiment": "negative", "text": state["text"], "response": ""}
    return {"sentiment": "neutral", "text": state["text"], "response": ""}

def respond_positive(state: RouterState) -> RouterState:
    """Respond to positive sentiment."""
    return {"response": "That's wonderful to hear! 😊", "text": state["text"], "sentiment": state["sentiment"]}

def respond_negative(state: RouterState) -> RouterState:
    """Respond to negative sentiment."""
    return {"response": "I'm sorry to hear that. How can I help?", "text": state["text"], "sentiment": state["sentiment"]}

def respond_neutral(state: RouterState) -> RouterState:
    """Respond to neutral sentiment."""
    return {"response": "I understand. Tell me more.", "text": state["text"], "sentiment": state["sentiment"]}

# %% [markdown]
#
# ## Router-Funktion
#
# Bestimmt den nächsten Node basierend auf State:

# %%
def route_by_sentiment(state: RouterState) -> str:
    """Route to appropriate response based on sentiment."""
    sentiment = state["sentiment"]
    if sentiment == "positive":
        return "respond_positive"
    elif sentiment == "negative":
        return "respond_negative"
    return "respond_neutral"

# %%
# TODO: Create graph with conditional edges

# %%

# %%

# %% [markdown]
#
# ## LangGraph mit LLMs
#
# Jetzt kombinieren wir LangGraph mit echten LLM-Aufrufen:

# %%
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Annotated
from langgraph.graph.message import add_messages

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/ministral-14b-2512",
    temperature=0.7
)

# %%
class ChatState(TypedDict):
    """State for chat with message history."""
    messages: Annotated[list, add_messages]

# %% [markdown]
#
# ## Einfacher Chatbot mit LangGraph

# %%
def chat_node(state: ChatState) -> ChatState:
    """Process messages with LLM."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# %%
# TODO: Create chat graph

# %%

# %% [markdown]
#
# ## Tool Calling mit LangGraph
#
# Agents können **Tools** nutzen, um Aufgaben zu erledigen:
#
# - Taschenrechner
# - Web-Suche
# - Datenbank-Abfragen
# - API-Aufrufe

# %%
from langchain_core.tools import tool

# %%
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python syntax."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def search_knowledge(query: str) -> str:
    """Search for information (simulated). Returns mock results."""
    knowledge_base = {
        "python": "Python is a high-level programming language known for readability.",
        "machine learning": "Machine learning is a subset of AI that learns from data.",
        "langchain": "LangChain is a framework for building LLM applications.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor LLM applications.",
    }
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    return "No information found for this query."

# %%

# %% [markdown]
#
# ## Agent mit Tools bauen
#
# Ein Agent entscheidet selbst, welches Tool er nutzt:

# %%
from langgraph.prebuilt import create_react_agent


# %%
# TODO: Create a ReAct agent with tools

# %%

# %%

# %% [markdown]
#
# ## Komplexeres Beispiel: Forschungsassistent
#
# Ein Agent, der:
# 1. Fragen analysiert
# 2. Informationen sucht
# 3. Eine Zusammenfassung erstellt

# %%
class ResearchState(TypedDict):
    """State for research assistant."""
    question: str
    search_results: str
    answer: str
    needs_more_info: bool

# %%
def analyze_question(state: ResearchState) -> ResearchState:
    """Analyze the question and decide if we need to search."""
    question = state["question"]
    # Simple heuristic: if question contains certain keywords, search
    needs_search = any(word in question.lower() for word in ["what", "how", "explain", "define"])
    return {
        "question": question,
        "search_results": "",
        "answer": "",
        "needs_more_info": needs_search
    }

def search_info(state: ResearchState) -> ResearchState:
    """Search for information (simulated)."""
    result = search_knowledge.invoke(state["question"])
    return {
        "question": state["question"],
        "search_results": result,
        "answer": "",
        "needs_more_info": False
    }

def generate_answer(state: ResearchState) -> ResearchState:
    """Generate answer based on search results."""
    context = state["search_results"] if state["search_results"] else "No additional context."
    prompt = f"Question: {state['question']}\nContext: {context}\nProvide a helpful answer:"

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "question": state["question"],
        "search_results": state["search_results"],
        "answer": response.content,
        "needs_more_info": False
    }

def route_search(state: ResearchState) -> str:
    """Route to search or directly to answer."""
    if state["needs_more_info"]:
        return "search"
    return "answer"

# %%
# TODO: Create research assistant graph

# %%

# %% [markdown]
#
# ## Graph Visualisierung
#
# LangGraph kann Graphs als Mermaid-Diagramme exportieren:

# %%

# %% [markdown]
#
# ## Zusammenfassung
#
# - **StateGraph**: Definiert Workflows mit Nodes und Edges
# - **State**: TypedDict für gemeinsamen Zustand
# - **Conditional Edges**: Verzweigungen basierend auf State
# - **Tools**: Agents können externe Funktionen aufrufen
# - **create_react_agent**: Einfache Agent-Erstellung
#
# **Nächster Schritt**: LangFlow für visuelle Workflows!

# %% [markdown]
#
# ## Workshop: Eigenen Agent bauen
#
# **Aufgabe**: Bauen Sie einen einfachen Assistenten, der:
#
# 1. Fragen zu Mathematik mit dem Calculator beantwortet
# 2. Wissensfragen mit der Suche beantwortet
# 3. Alles andere direkt beantwortet
#
# **Bonus**: Fügen Sie ein eigenes Tool hinzu!

# %%
# Workshop space - try building your own agent!

# %%

# %%

# %%

# %%
