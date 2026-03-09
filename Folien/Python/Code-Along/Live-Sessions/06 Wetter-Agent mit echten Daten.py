# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Wetter-Agent mit echten Daten</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Von Fake-Daten zu echten APIs
#
# - Im letzten Notebook: zufällige Wetterdaten
# - Jetzt: echte Wetterdaten aus dem Internet
# - **Open-Meteo**: Kostenlose Wetter-API (kein API-Key nötig)
# - **Nominatim**: Kostenlose Geocoding-API (kein API-Key nötig)

# %%
# !pip install langchain langgraph langchain-openai python-dotenv requests --root-user-action ignore

# %%
import os
import time
from dataclasses import dataclass
from textwrap import wrap

import requests
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
# ## Nominatim-Konfiguration
#
# - Öffentliche API: Rate-Limit von 1 Anfrage/Sekunde
# - Lokale Docker-Instanz: Kein Rate-Limit
# - Konfigurierbar über Umgebungsvariable `NOMINATIM_URL`

# %%
NOMINATIM_BASE_URL = os.getenv(
    "NOMINATIM_URL", "https://nominatim.openstreetmap.org"
)
NOMINATIM_RATE_LIMIT = NOMINATIM_BASE_URL.startswith(
    "https://nominatim.openstreetmap.org"
)

# %% [markdown]
#
# ## Geocoding mit Nominatim
#
# - Übersetzt Städtenamen in Breitengrad/Längengrad
# - Benötigt einen `User-Agent`-Header
# - Ergebnis: `(lat, lon)` Tupel

# %%
def geocode_location(location: str) -> tuple[float, float]:
    """Convert a location name to (latitude, longitude)."""
    response = requests.get(
        f"{NOMINATIM_BASE_URL}/search",
        params={"q": location, "format": "json", "limit": 1},
        headers={"User-Agent": "tourist-info-agent/1.0"},
    )
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f"Location not found: {location}")
    if NOMINATIM_RATE_LIMIT:
        time.sleep(1)
    return float(data[0]["lat"]), float(data[0]["lon"])

# %% [markdown]
#
# ### Geocoding testen

# %%

# %% [markdown]
#
# ## Open-Meteo: Kostenlose Wetter-API
#
# - Nimmt Breitengrad/Längengrad als Parameter
# - Liefert aktuelle Wetterdaten (Temperatur, Wind, etc.)
# - Kein API-Key erforderlich
# - WMO-Wettercodes für Beschreibungen

# %%
WMO_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy",
    3: "overcast", 45: "foggy", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers", 95: "thunderstorm",
}

# %%
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    pass

# %% [markdown]
#
# ### Wetter-Tool testen

# %%

# %% [markdown]
#
# ## Kontext und Agent aufbauen
#
# - Gleiche Architektur wie im einfachen Agenten
# - `WeatherContext` für Benutzer-Standort
# - Jetzt mit echten Wetterdaten!

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

# %%
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can
tell from the question that they mean wherever they are, use the
get_user_location tool to find their location.
"""

# %%
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: str | None = None

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

# %%
checkpointer = InMemorySaver()

# %%
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather, get_user_location],
    context_schema=WeatherContext,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

# %% [markdown]
#
# ## Den Agenten testen
#
# - Echte Wetterdaten für eine echte Stadt
# - Der Agent ruft automatisch die richtigen Tools auf

# %%
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
context = WeatherContext(location="Vienna", user_id="user_123")

# %%
response = agent.invoke(
    input=HumanMessage(content="What is the weather outside?"),
    context=context,
    config=config,
)

# %%

# %%

# %% [markdown]
#
# ### Nachrichtenverlauf inspizieren

# %%
