# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Touristen-Info Agent: Mehr Tools</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Vom Wetter-Agenten zum Reise-Assistenten
#
# - Ein Tool ist nützlich, mehrere Tools machen einen echten Assistenten
# - Neue Tools für unseren Touristen-Info-Agenten:
#   - **Länder-Informationen** (RestCountries API)
#   - **Sehenswürdigkeiten suchen** (Nominatim)
#   - **Entfernungen berechnen** (Haversine-Formel)

# %%
# !pip install langchain langgraph langchain-openai python-dotenv requests --root-user-action ignore

# %%
import math
import os
import time
from dataclasses import dataclass
import re
from textwrap import fill

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
def format_llm_output(text, width=72):
    """Wrap long lines while preserving existing line breaks and list structure."""
    result = []
    for line in text.split("\n"):
        if not line.strip():
            result.append("")
        else:
            m = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)", line)
            indent = " " * len(m.group(1)) if m else ""
            result.append(fill(line, width=width, subsequent_indent=indent))
    return "\n".join(result)


# %%
def print_llm_output(text, width=72):
    """Print LLM output with proper formatting."""
    print(format_llm_output(text, width=width))


# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NOMINATIM_BASE_URL = os.getenv(
    "NOMINATIM_URL", "https://nominatim.openstreetmap.org"
)
NOMINATIM_RATE_LIMIT = NOMINATIM_BASE_URL.startswith(
    "https://nominatim.openstreetmap.org"
)

# %%
def geocode_location(location: str) -> tuple[float, float]:
    """Convert a location name to (latitude, longitude)."""
    response = requests.get(
        f"{NOMINATIM_BASE_URL}/search",
        params={"q": location, "format": "json", "limit": 1},
        headers={"User-Agent": "tourist-info-agent/1.0"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f"Location not found: {location}")
    if NOMINATIM_RATE_LIMIT:
        time.sleep(1)
    return float(data[0]["lat"]), float(data[0]["lon"])

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
    try:
        lat, lon = geocode_location(location)
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,"
                           "weather_code,wind_speed_10m",
            },
            timeout=10,
        )
        response.raise_for_status()
        current = response.json()["current"]
        weather_desc = WMO_WEATHER_CODES.get(
            current["weather_code"], "unknown"
        )
        return (
            f"Weather in {location}: {weather_desc}, "
            f"{current['temperature_2m']}°C, "
            f"wind {current['wind_speed_10m']} km/h, "
            f"humidity {current['relative_humidity_2m']}%"
        )
    except Exception as e:
        return f"Error getting weather for {location}: {e}"

# %% [markdown]
#
# ## Tool 1: Länder-Informationen
#
# - **RestCountries API**: Kostenlos, kein API-Key
# - Liefert: Hauptstadt, Bevölkerung, Währungen, Sprachen, Region
# - Endpunkt: `restcountries.com/v3.1/name/{name}`

# %%
# @tool
# def get_country_info(country: str) -> str:
#     """Get general information about a country (capital, population,
#     currencies, languages, region)."""
#     pass

# %% [markdown]
#
# ### Länder-Info testen

# %%

# %% [markdown]
#
# ## Tool 2: Sehenswürdigkeiten suchen
#
# - Nominatim `/search` mit Freitext-Suche
# - Kombiniert Kategorie und Ort (z.B. "museum near Paris")
# - Liefert die 5 besten Ergebnisse mit Name und Adresse

# %%
# @tool
# def search_points_of_interest(location: str, category: str) -> str:
#     """Search for points of interest (e.g., restaurants, museums, parks)
#     near a given location. Returns up to 5 results."""
#     pass

# %% [markdown]
#
# ### Sehenswürdigkeiten testen

# %%

# %% [markdown]
#
# ## Tool 3: Entfernungen berechnen
#
# - **Haversine-Formel**: Entfernung auf einer Kugel
# - Keine API nötig — reine Mathematik
# - Schätzt auch Reisezeit (Auto ~80 km/h, Zug ~120 km/h)

# %%
# @tool
# def calculate_distance(city_a: str, city_b: str) -> str:
#     """Calculate the distance between two cities in kilometers and
#     estimate travel times by car and train."""
#     pass

# %% [markdown]
#
# ### Entfernungen testen

# %%

# %% [markdown]
#
# ## Den Touristen-Info-Agenten zusammenbauen
#
# - Neuer System-Prompt: Reiseberater statt Wetter-Clown
# - Neues Antwort-Format mit Reisetipps
# - Alle 5 Tools kombiniert

# %%
@dataclass
class TouristContext:
    """Runtime context for the tourist info agent."""
    user_id: str
    location: str

# %%
@tool
def get_user_location(runtime: ToolRuntime[TouristContext]) -> str:
    """Get the user's current location."""
    return runtime.context.location

# %%
TOURIST_SYSTEM_PROMPT = """You are a helpful tourist information assistant. \
You help travelers plan their trips by providing weather forecasts, country \
facts, points of interest, and travel distances.

You have access to these tools:

- get_weather: Get current weather for a location
- get_user_location: Get the user's current location
- get_country_info: Get facts about a country (capital, population, etc.)
- search_points_of_interest: Find attractions, restaurants, museums, etc.
- calculate_distance: Calculate distance and travel time between two cities

Be friendly and informative. When a user asks about a destination, proactively \
offer relevant information from multiple tools when appropriate.
"""

# %%
@dataclass
class TouristResponse:
    """Response schema for the tourist info agent."""
    response: str
    travel_tips: str | None = None

# %%
MODEL_NAME = "openai/gpt-5.4"

# %%
model = init_chat_model(
    model=MODEL_NAME,
    model_provider="openai",
    base_url=OPENROUTER_BASE_URL,
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0.0,
    timeout=30,
    max_tokens=2048,
)

# %%
checkpointer = InMemorySaver()

# %%
tourist_agent = create_agent(
    model=model,
    system_prompt=TOURIST_SYSTEM_PROMPT,
    tools=[
        get_weather,
        get_user_location,
        get_country_info,
        search_points_of_interest,
        calculate_distance,
    ],
    context_schema=TouristContext,
    response_format=ToolStrategy(TouristResponse),
    checkpointer=checkpointer,
)

# %% [markdown]
#
# ## Den Agenten testen
#
# - Verschiedene Anfragen zeigen, wie der Agent die richtigen Tools
#   auswählt

# %%
config: RunnableConfig = {"configurable": {"thread_id": "1"}, "recursion_limit": 10}
context = TouristContext(location="Berlin", user_id="user_123")

# %% [markdown]
#
# ### Wetter-Anfrage

# %%
response = tourist_agent.invoke(
    input={"messages": [HumanMessage(content="What's the weather like in Rome?")]},
    context=context,
    config=config,
)

# %%

# %% [markdown]
#
# ### Länder-Anfrage

# %%
config2: RunnableConfig = {"configurable": {"thread_id": "2"}, "recursion_limit": 10}

# %%
response2 = tourist_agent.invoke(
    input={"messages": [HumanMessage(content="Tell me about Italy.")]},
    context=context,
    config=config2,
)

# %%

# %% [markdown]
#
# ### Entfernungs-Anfrage

# %%
config3: RunnableConfig = {"configurable": {"thread_id": "3"}, "recursion_limit": 10}

# %%
response3 = tourist_agent.invoke(
    input={"messages": [HumanMessage(
        content="How far is it from Berlin to Munich?"
    )]},
    context=context,
    config=config3,
)

# %%

# %% [markdown]
#
# ### Sehenswürdigkeiten-Anfrage

# %%
config4: RunnableConfig = {"configurable": {"thread_id": "4"}, "recursion_limit": 10}

# %%
response4 = tourist_agent.invoke(
    input={"messages": [HumanMessage(content="Find me museums near Paris.")]},
    context=context,
    config=config4,
)

# %%

# %% [markdown]
#
# ## Reisetipps

# %%

# %%
