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

# %%
from langgraph.graph import StateGraph, START, END

# %% [markdown]
#
# ## Wann Graphs statt Chains?
#
# - **Bedingte Logik**: Verschiedene Pfade je nach Input
# - **Schleifen**: Wiederhole Schritte bis Bedingung erfüllt
# - **Agentic Workflows**: LLM entscheidet nächsten Schritt
# - **Komplexe Workflows**: Mehrere parallele Prozesse

# %% [markdown]
#
# ## Zusammenfassung
#
# - LangGraph für komplexe Workflows
# - Nodes und Edges definieren Ablauf
# - Conditional Edges für Entscheidungen
# - State Management für Datenweitergabe
#
# **Nächster Schritt**: LangFlow für visuelle Workflows!

# %%
