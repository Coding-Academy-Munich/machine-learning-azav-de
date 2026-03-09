# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einfacher Wetter-Agent</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Was ist ein Agent?
#
# - **LLM + Tools + Entscheidungslogik**
# - Das LLM entscheidet, welche Tools es aufruft
# - Tools liefern Daten aus der realen Welt
# - Der Agent kombiniert die Ergebnisse zu einer Antwort

# %% [markdown]
#
# ## Unser Plan
#
# - Einen Wetter-Agenten bauen
# - `@tool` Decorator für Tool-Definitionen
# - `ToolRuntime` für Kontext-Informationen
# - `create_agent` für die Agent-Erstellung
# - `ToolStrategy` für strukturierte Ausgaben

# %%
# !pip install langchain langgraph langchain-openai python-dotenv --root-user-action ignore

# %%
import os
import random
from dataclasses import dataclass
from textwrap import wrap

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

# %%
load_dotenv()

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# %% [markdown]
#
# ## Das LLM konfigurieren
#
# - `init_chat_model` erstellt ein Chat-Modell
# - Wir verwenden OpenRouter als Provider
# - Modell: `openai/gpt-4.1-nano` (günstig, gutes Tool-Calling)

# %%
model = init_chat_model(
    model="openai/gpt-4.1-nano",
    model_provider="openai",
    base_url=OPENROUTER_BASE_URL,
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0.5,
    timeout=10,
    max_tokens=2048,
)

# %% [markdown]
#
# ## Tools definieren
#
# - `@tool` macht eine Python-Funktion zum Agent-Tool
# - Docstring wird als Tool-Beschreibung verwendet
# - Das LLM entscheidet anhand der Beschreibung, wann es das Tool nutzt

# %%
WEATHER_CHOICES = [
    "it's 19 degrees and sunny",
    "it's 10 degrees and raining",
    "it's 25 degrees with scattered clouds",
]

# %%
@tool
def get_weather_for_location(location: str) -> str:
    """Get the weather for a specific location."""
    return random.choice(WEATHER_CHOICES)

# %% [markdown]
#
# ## Kontext mit `ToolRuntime`
#
# - `ToolRuntime` gibt Tools Zugriff auf Laufzeit-Kontext
# - Kontext enthält Informationen über den Benutzer
# - Definiert über ein eigenes `dataclass`

# %%
@dataclass
class WeatherContext:
    """Custom runtime context schema."""
    user_id: str
    location: str

# %%
@tool
def get_user_location(runtime: ToolRuntime[WeatherContext]) -> str:
    """Get the user's current location."""
    return runtime.context.location

# %% [markdown]
#
# ## System-Prompt und Antwort-Format
#
# - System-Prompt definiert die Persönlichkeit des Agenten
# - `ResponseFormat` definiert die Struktur der Antwort
# - `ToolStrategy` stellt sicher, dass die Antwort dem Format entspricht

# %%
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's current location

If a user asks you about the weather, you should first get their location using
get_user_location, and then get the weather for that location using
get_weather_for_location. You should then respond with a pun about the weather.
"""

# %%
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: str | None = None

# %% [markdown]
#
# ## Den Agenten erstellen
#
# - `create_agent` kombiniert Modell, Tools und Konfiguration
# - `InMemorySaver` ermöglicht Konversationsgedächtnis
# - Der Agent entscheidet selbstständig, welche Tools er aufruft

# %%
checkpointer = InMemorySaver()

# %%
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather_for_location, get_user_location],
    context_schema=WeatherContext,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

# %% [markdown]
#
# ## Den Agenten aufrufen
#
# - `config` identifiziert die Konversation (Thread-ID)
# - `context` liefert Laufzeit-Informationen (Standort, User-ID)
# - `HumanMessage` enthält die Benutzer-Nachricht

# %%
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
context = WeatherContext(location="New York", user_id="user_123")

# %%
weather_input = HumanMessage(
    content="Hello! Can you tell me what the weather is like today?"
)

# %%
response = agent.invoke(
    input=weather_input,
    context=context,
    config=config,
)

# %% [markdown]
#
# ## Die Antwort auswerten
#
# - `structured_response` enthält die strukturierte Antwort
# - `messages` enthält den gesamten Nachrichtenverlauf inkl. Tool-Aufrufe

# %%

# %%

# %%
