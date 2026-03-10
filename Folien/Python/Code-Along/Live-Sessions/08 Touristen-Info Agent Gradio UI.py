# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Touristen-Info Agent: Gradio UI</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Agent + Gradio = Chat-Anwendung
#
# - Bisher: Agent-Aufrufe im Notebook
# - Jetzt: Interaktive Chat-Oberfläche mit Gradio
# - Gradio `ChatInterface` kennen Sie bereits aus dem LLM-Chatbot
# - Herausforderung: Agent-Messages ↔ Gradio-Format verbinden

# %%
# !pip install langchain langgraph langchain-openai python-dotenv requests gradio --root-user-action ignore

# %%
import math
import os
import time
import uuid
from dataclasses import dataclass

import gradio as gr
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
NOMINATIM_BASE_URL = os.getenv(
    "NOMINATIM_URL", "https://nominatim.openstreetmap.org"
)
NOMINATIM_RATE_LIMIT = NOMINATIM_BASE_URL.startswith(
    "https://nominatim.openstreetmap.org"
)

# %% [markdown]
#
# ## Tools und Agent definieren
#
# - Gleiche Tools wie im vorherigen Notebook
# - Alles in einem Block, damit das Notebook eigenständig funktioniert

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


@tool
def get_country_info(country: str) -> str:
    """Get general information about a country (capital, population,
    currencies, languages, region)."""
    try:
        response = requests.get(
            f"https://restcountries.com/v3.1/name/{country}",
            params={"fields": "name,capital,population,currencies,"
                              "languages,region,subregion"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()[0]
        currencies = ", ".join(
            f"{v['name']} ({v.get('symbol', '?')})"
            for v in data.get("currencies", {}).values()
        )
        languages = ", ".join(data.get("languages", {}).values())
        capital = ", ".join(data.get("capital", ["unknown"]))
        return (
            f"{data['name']['common']}: "
            f"Capital: {capital}, "
            f"Population: {data['population']:,}, "
            f"Region: {data.get('region', '?')} ({data.get('subregion', '?')}), "
            f"Currencies: {currencies}, "
            f"Languages: {languages}"
        )
    except Exception as e:
        return f"Error getting country info for {country}: {e}"

# %%
@tool
def search_points_of_interest(location: str, category: str) -> str:
    """Search for points of interest (e.g., restaurants, museums, parks)
    near a given location. Returns up to 5 results."""
    try:
        response = requests.get(
            f"{NOMINATIM_BASE_URL}/search",
            params={
                "q": f"{category} near {location}",
                "format": "json",
                "limit": 5,
                "addressdetails": 1,
            },
            headers={"User-Agent": "tourist-info-agent/1.0"},
            timeout=10,
        )
        response.raise_for_status()
        if NOMINATIM_RATE_LIMIT:
            time.sleep(1)
        results = response.json()
        if not results:
            return f"No {category} found near {location}."
        lines = []
        for r in results:
            name = r.get("display_name", "Unknown")
            lines.append(f"- {name}")
        return f"{category.title()} near {location}:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error searching {category} near {location}: {e}"


@tool
def calculate_distance(city_a: str, city_b: str) -> str:
    """Calculate the distance between two cities in kilometers and
    estimate travel times by car and train."""
    try:
        lat1, lon1 = geocode_location(city_a)
        lat2, lon2 = geocode_location(city_b)
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        distance = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        car_hours = distance / 80
        train_hours = distance / 120
        return (
            f"Distance from {city_a} to {city_b}: {distance:.0f} km. "
            f"Estimated travel time by car: {car_hours:.1f}h, "
            f"by train: {train_hours:.1f}h."
        )
    except Exception as e:
        return f"Error calculating distance from {city_a} to {city_b}: {e}"

# %%
@dataclass
class TouristContext:
    """Runtime context for the tourist info agent."""
    user_id: str
    location: str


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
model = init_chat_model(
    model="openai/gpt-4.1-nano",
    model_provider="openai",
    base_url=OPENROUTER_BASE_URL,
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0.5,
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
# ## Gradio-Callback schreiben
#
# - Gradio gibt `(message, history)` an die Callback-Funktion
# - Wir erstellen eine `HumanMessage` und rufen den Agenten auf
# - Jede Konversation bekommt eine eigene `thread_id`

# %%
thread_id = str(uuid.uuid4())
context = TouristContext(location="Berlin", user_id="user_123")

# %%
def respond(message, history):
    """Gradio callback that invokes the tourist info agent."""
    pass

# %% [markdown]
#
# ## Chat-Interface erstellen
#
# - `gr.ChatInterface`: Einfache Chat-Oberfläche
# - Titel und Beschreibung für den Benutzer
# - Beispiel-Nachrichten für den Einstieg

# %%
chatbot = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Tourist Info Agent",
    description="Ask me about weather, countries, points of interest, "
                "or travel distances!",
    examples=[
        "What's the weather like in Tokyo?",
        "Tell me about Spain.",
        "Find me restaurants near Rome.",
        "How far is it from London to Paris?",
    ],
)

# %%
# chatbot.launch()

# %% [markdown]
#
# ## Ideen zur Erweiterung
#
# - **Streaming**: Antworten Wort für Wort anzeigen
# - **Mehr Tools**: Flugsuche, Hotelpreise, Währungsumrechnung
# - **Personalisierung**: Benutzer-Präferenzen speichern
# - **Deployment**: Auf HuggingFace Spaces veröffentlichen

# %% [markdown]
#
# ## Zusammenfassung
#
# - **Slide 1**: Einfacher Agent mit `@tool` und `create_agent`
# - **Slide 2**: Echte Wetterdaten mit Open-Meteo und Nominatim
# - **Slide 3**: Multi-Tool-Agent mit 5 verschiedenen Werkzeugen
# - **Slide 4**: Interaktive Chat-Oberfläche mit Gradio
# - Alle APIs kostenlos und ohne API-Key nutzbar
