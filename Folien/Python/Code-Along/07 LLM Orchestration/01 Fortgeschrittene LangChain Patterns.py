# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Fortgeschrittene LangChain Patterns</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was wir bisher können
#
# - Einfache LLM-Aufrufe
# - Chatbots mit Memory
# - RAG-Systeme
#
# **Jetzt**: Komplexe Multi-Step Workflows!

# %%
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# %% [markdown]
#
# ## Sequential Chains mit LCEL
#
# - Mehrere Schritte nacheinander mit Pipe-Operator (`|`)
# - Output von Schritt 1 → Input für Schritt 2
# - **Beispiel**: Idee generieren → Story ausarbeiten
# - Nutzt LangChain Expression Language (LCEL)

# %%
# TODO: Create sequential chain using LCEL

# %%

# %% [markdown]
#
# ## LangChain Agents (Fortgeschritten)
#
# - Agents können **Entscheidungen treffen**
# - Nutzen **Tools** (Taschenrechner, Suche, APIs)
# - Planen Schritte selbst
# - **Moderne Implementierung**: LangGraph (siehe weiterführende Kurse)
# - **Einfachere Alternative**: Strukturierte Chains mit LCEL

# %% [markdown]
#
# ## Zusammenfassung
#
# - **LCEL** (Pipe-Operator `|`) für mehrstufige Verarbeitung
# - Chains lassen sich elegant verketten
# - Output-Parser für strukturierte Ergebnisse
# - Error Handling ist wichtig
#
# **Für komplexe Agents**: LangGraph (fortgeschrittenes Thema)

# %%
