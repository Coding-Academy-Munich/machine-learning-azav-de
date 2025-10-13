# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Requests Beispiele</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# - Dieses Notebook zeigt die Verwendung der Python-Bibliothek `requests`, um
#   mit verschiedenen freien und offenen Webdiensten zu interagieren.
# - Jeder Abschnitt stellt eine API vor, definiert ihre URL, führt eine Anfrage
#   durch und zeigt die JSON-Antwort.

# %%
# !pip install requests --root-user-action ignore

# %%
import requests

# %% [markdown]
#
# ## 🌐 1. JSONPlaceholder
#
# - JSONPlaceholder ist eine Fake-Online-REST-API, die sich perfekt zum Testen
#   und Entwickeln von Prototypen eignet.
# - Sie bietet typische API-Endpunkte (wie `/users`, `/posts`, `/comments`) und
#   unterstützt alle gängigen HTTP-Methoden.

# %%
# Define the base URL for the service
jsonplaceholder_url = "https://jsonplaceholder.typicode.com"

# %% [markdown]
#
# ### Eine Liste von Ressourcen abrufen
#
# Hier rufen wir die ersten 5 Beiträge ab.

# %%
# Make a GET request to the /posts endpoint with a parameter to limit results
response = requests.get(f"{jsonplaceholder_url}/posts", params={"_limit": 5})

# %%
# Display the JSON response from the server
response.json()

# %% [markdown]
#
# ### Eine einzelne Ressource abrufen
#
# Nun rufen wir einen einzelnen Beitrag anhand seiner ID ab (in diesem Fall Beitrag #42).

# %%
post_id = 42

# %%
response = requests.get(f"{jsonplaceholder_url}/posts/{post_id}")

# %%
response.json()

# %% [markdown]
#
# ### 📬 2. HTTPBin
#
# - HTTPBin ist ein Dienst für Anfragen und Antworten.
# - Es fungiert wie ein Spiegel und gibt Informationen über die gesendete
#   Anfrage zurück.
# - Es ist nützlich zum Debuggen und Verstehen, wie Ihre HTTP-Anfragen aussehen.

# %%
# Define the base URL for the service
httpbin_url = "https://httpbin.org"

# %% [markdown]
#
# ### Eine POST-Anfrage inspizieren
#
# - Wir senden eine JSON-Payload in einer POST-Anfrage.
# - HTTPBin gibt die gesendeten Daten in seiner Antwort zurück.

# %%
# The data we want to send
payload = {"student_name": "Alex", "grade": "A+", "course_id": 101}

# %%
# Make a POST request, sending our payload as JSON
response = requests.post(f"{httpbin_url}/post", json=payload)

# %%
response.json()

# %% [markdown]
#
# ## 🛰️ 3. Open Notify (ISS-Standort)
#
# - Open Notify bietet eine einfache API für den Zugriff auf einige Daten von
#   NASA.
# - Der beliebteste Endpunkt gibt den Echtzeitstandort der Internationalen
#   Raumstation (ISS) zurück.
# - Das ist ein Beispiel für eine Live-Datenquelle.

# %%
# Define the full URL for the endpoint
iss_now_url = "http://api.open-notify.org/iss-now.json"

# %%
# Make a GET request to find the ISS (can be very unreliable)
# response = requests.get(iss_now_url)

# %%
# response.json()

# %%
import time

# %%
# for _ in range(3):
#     response = requests.get(iss_now_url)
#     position = response.json()['iss_position']
#     print(f"Latitude: {position['latitude']}, Longitude: {position['longitude']}")
#     time.sleep(1)


# %% [markdown]
#
# ## 🎮 4. PokéAPI
#
# - Die PokéAPI ist eine unterhaltsame, kostenlose und umfassende RESTful-API
#   für alles rund um Pokémon.
# - Sie ist ein großartiges Beispiel für eine gut strukturierte API, die
#   Pfadparameter verwendet, um spezifische Ressourcen zu identifizieren.

# %%
# Define the base URL for the service
pokeapi_url = "https://pokeapi.co/api/v2"

# %% [markdown]
#
# ### Daten für ein spezifisches Pokémon abrufen
#
# Wir können detaillierte Informationen über ein Pokémon abrufen, indem wir
# seinen Namen zum URL-Pfad hinzufügen.

# %%
pokemon_name = "ditto"

# %%
# Make a GET request for a specific pokémon
response = requests.get(f"{pokeapi_url}/pokemon/{pokemon_name}")

# %%
response.json()

# %% [markdown]
#
# ### Eine Liste von Pokémon abrufen (mit Paginierung)
#
# - Die API ermöglicht es auch, eine Liste aller Pokémon abzurufen.
# - Die Ergebnisse sind **paginierbar**, das heißt, sie werden in Seiten
#   geliefert.
# - Sie können die Parameter `limit` und `offset` verwenden, um zu steuern,
#   wie viele Sie erhalten und auf welcher Seite Sie beginnen möchten.

# %%
# Let's get the first 10 Pokémon from the list.
list_params = {'limit': 10}

# %%
response = requests.get(f"{pokeapi_url}/pokemon", params=list_params)

# %% [markdown]
#
# - Achten Sie auf die Struktur der Antwort.
# - Die tatsächliche Liste der Pokémon befindet sich im Schlüssel
#   **`"results"`**.
# - Die API stellt auch
#   - die Gesamtanzahl **`"count"`** aller Pokémon und
#   - eine URL im Schlüssel **`"next"`** zur Verfügung, um die nächste Seite der
#     Ergebnisse abzurufen.

# %%
response.json()

# %% [markdown]
#
# Anfordern der nächsten Seite der Ergebnisse

# %%
next_url = response.json().get('next')
next_url

# %%
if next_url:
    response = requests.get(next_url)

# %%
response.json()

# %%
