# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Modellvergleich: Touristen-Info Agent</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>
#
# <div style="text-align:center;">Coding-Akademie München</div>
# <br/>


# %% [markdown]
#
# ## Warum Modelle vergleichen?
#
# - Gleiche Agent-Architektur, verschiedene LLMs
# - Unterschiede in Antwortqualität, Tool-Nutzung, Geschwindigkeit
# - LangChain abstrahiert den Provider-Wechsel
# - Vergleich hilft bei der Modellauswahl für Produktionssysteme

# %% [markdown]
#
# ### Was wir vergleichen
#
# - **GPT-4.1** (OpenAI): Starke Tool-Unterstützung, schnell
# - **Claude Sonnet** (Anthropic): Ausführliche, gut strukturierte Antworten
# - **Mistral Medium** (Mistral AI): Europäisches Open-Weight-Modell
# - Alle über **OpenRouter** mit einem API-Key

# %%
# !pip install langchain langgraph langchain-openai langchain-anthropic python-dotenv requests gradio --root-user-action ignore

# %%
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import gradio as gr
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

# %%
load_dotenv()

# %% [markdown]
#
# ## Modell-Konfiguration
#
# - Jedes Modell als Dictionary mit Name, Modell-ID, Provider
# - OpenRouter als einheitlicher Endpunkt für alle Modelle
# - Einfach erweiterbar: neues Modell = neuer Dictionary-Eintrag

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NOMINATIM_BASE_URL = os.getenv(
    "NOMINATIM_URL", "https://nominatim.openstreetmap.org"
)
NOMINATIM_RATE_LIMIT = NOMINATIM_BASE_URL.startswith(
    "https://nominatim.openstreetmap.org"
)

# %%
MODEL_CONFIGS = [
    {
        "name": "GPT-4.1",
        "model": "openai/gpt-4.1",
        "model_provider": "openai",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
    {
        "name": "Claude Sonnet",
        "model": "anthropic/claude-sonnet-4",
        "model_provider": "openai",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
    {
        "name": "Mistral Medium",
        "model": "mistralai/mistral-medium-3",
        "model_provider": "openai",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
    },
    # Native API configs (require separate API keys):
    # {
    #     "name": "Claude Sonnet (native)",
    #     "model": "claude-sonnet-4-6-20250514",
    #     "model_provider": "anthropic",
    #     "api_key_env": "ANTHROPIC_API_KEY",
    # },
    # {
    #     "name": "GPT-4.1 (native)",
    #     "model": "gpt-4.1",
    #     "model_provider": "openai",
    #     "api_key_env": "OPENAI_API_KEY",
    # },
    # Open-source models via OpenRouter:
    # {
    #     "name": "Llama 4 Scout",
    #     "model": "meta-llama/llama-4-scout",
    #     "model_provider": "openai",
    #     "base_url": OPENROUTER_BASE_URL,
    #     "api_key_env": "OPENROUTER_API_KEY",
    # },
]

# %% [markdown]
#
# ## Tools definieren
#
# - Gleiche 5 Tools wie im vorherigen Notebook
# - Wetter, Länder-Info, Sehenswürdigkeiten, Entfernungen, Standort
# - Die Tools sind identisch — nur das LLM ändert sich

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
ALL_TOOLS = [
    get_weather,
    get_user_location,
    get_country_info,
    search_points_of_interest,
    calculate_distance,
]

# %% [markdown]
#
# ## Agent-Fabrik
#
# - Eine Funktion erstellt einen kompletten Agenten aus einer Konfiguration
# - Jedes Modell bekommt eigenen Checkpointer und Thread-ID
# - Kein `ToolStrategy` — nicht alle Modelle unterstützen strukturierte
#   Ausgaben zuverlässig

# %%
def create_tourist_agent(config):
    """Create a tourist info agent from a model configuration."""
    model = init_chat_model(
        model=config["model"],
        model_provider=config["model_provider"],
        base_url=config.get("base_url"),
        api_key=os.environ[config["api_key_env"]],
        temperature=0.0,
        timeout=60,
        max_tokens=2048,
    )
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        system_prompt=TOURIST_SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        context_schema=TouristContext,
        checkpointer=checkpointer,
    )
    return {
        "agent": agent,
        "name": config["name"],
        "thread_id": str(uuid.uuid4()),
    }

# %%
context = TouristContext(location="Berlin", user_id="user_123")

# %%
agents = [create_tourist_agent(cfg) for cfg in MODEL_CONFIGS]

# %% [markdown]
#
# ## Sequenzieller Vergleich
#
# - Gleiche Anfrage an jedes Modell senden
# - Antworten nacheinander anzeigen
# - Gut für Notebooks, einfach zu verstehen

# %%
def query_single_model(agent_info, message):
    """Query one model and return the response text or error."""
    try:
        config: RunnableConfig = {
            "configurable": {"thread_id": agent_info["thread_id"]},
            "recursion_limit": 10,
        }
        result = agent_info["agent"].invoke(
            input={"messages": [HumanMessage(content=message)]},
            context=context,
            config=config,
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

# %%
question = "What's the weather like in Rome?"

# %%
for agent_info in agents:
    print(f"=== {agent_info['name']} ===")
    response = query_single_model(agent_info, question)
    print(response)
    print()

# %% [markdown]
#
# ## Parallele Ausführung
#
# - API-Aufrufe sind I/O-gebunden — Threads funktionieren gut
# - `ThreadPoolExecutor` aus der Standardbibliothek
# - Alle Modelle gleichzeitig abfragen, Ergebnisse sammeln

# %%
def compare_models(message):
    """Query all models in parallel and return results."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {
            executor.submit(query_single_model, agent_info, message): agent_info["name"]
            for agent_info in agents
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
    return results

# %%
results = compare_models("Tell me about Japan.")

# %%
for name, response in results.items():
    print(f"=== {name} ===")
    print(response)
    print()

# %% [markdown]
#
# ## Gradio: Vergleichs-UI
#
# - Einzelnes Eingabefeld, Antworten nebeneinander
# - `gr.Blocks` statt `ChatInterface` für flexibles Layout
# - Ein `gr.Chatbot` pro Modell in eigener Spalte

# %%
def build_comparison_ui(agents_list):
    """Build a Gradio comparison UI for multiple tourist info agents."""
    with gr.Blocks(title="Tourist Info Agent: Model Comparison") as demo:
        gr.Markdown("# Tourist Info Agent: Model Comparison")
        gr.Markdown(
            "Ask a travel question and compare how different models respond."
        )
        with gr.Row():
            user_input = gr.Textbox(
                label="Your question",
                placeholder="e.g., What's the weather like in Tokyo?",
                scale=4,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        chatbots = []
        with gr.Row():
            for agent_info in agents_list:
                with gr.Column():
                    chatbots.append(
                        gr.Chatbot(
                            label=agent_info["name"],
                            type="messages",
                            height=400,
                        )
                    )

        def respond(message, *histories):
            if not message.strip():
                return [""] + list(histories)
            results = {}
            with ThreadPoolExecutor(max_workers=len(agents_list)) as executor:
                futures = {
                    executor.submit(
                        query_single_model, agent_info, message
                    ): i
                    for i, agent_info in enumerate(agents_list)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
            new_histories = []
            for i, hist in enumerate(histories):
                new_hist = list(hist) + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": results[i]},
                ]
                new_histories.append(new_hist)
            return [""] + new_histories

        submit_btn.click(
            fn=respond,
            inputs=[user_input] + chatbots,
            outputs=[user_input] + chatbots,
        )
        user_input.submit(
            fn=respond,
            inputs=[user_input] + chatbots,
            outputs=[user_input] + chatbots,
        )

        gr.Examples(
            examples=[
                "What's the weather like in Tokyo?",
                "Tell me about Spain.",
                "Find me museums near Paris.",
                "How far is it from London to Paris?",
            ],
            inputs=user_input,
        )

    return demo

# %%
demo = build_comparison_ui(agents)

# %%
# demo.launch()

# %% [markdown]
#
# ## Beobachtungen und Diskussion
#
# - **Antwortqualität**: Welches Modell gibt die hilfreichsten Antworten?
# - **Tool-Nutzung**: Rufen alle Modelle die gleichen Tools auf?
# - **Geschwindigkeit**: Welches Modell antwortet am schnellsten?
# - **Fehler**: Scheitert ein Modell beim Tool-Calling?
# - **Stil**: Wie unterscheidet sich der Tonfall?

# %% [markdown]
#
# ### Nächste Schritte
#
# - Mehr Modelle hinzufügen (Llama, Qwen, Gemini)
# - Native APIs testen (Anthropic, OpenAI) statt OpenRouter
# - Metriken sammeln: Latenz, Token-Verbrauch, Kosten
# - Automatisierte Bewertung mit einem Evaluierungs-Framework

# %%
