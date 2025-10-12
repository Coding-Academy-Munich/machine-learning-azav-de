# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Das requests Package</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Das `requests` Package
#
# - Python-Bibliothek für HTTP-Anfragen
# - Macht Web Service-Aufrufe sehr einfach
# - Muss installiert werden: `pip install requests`

# %% [markdown]
#
# ## Vorteile von `requests`
#
# - Konvertiert automatisch zwischen JSON und Python
# - Einfache Fehlerbehandlung
# - Intuitive API
# - Weit verbreitet und gut dokumentiert

# %% [markdown]
#
# ## GET Requests: Daten abrufen
#
# Wir verwenden die [JSON Placeholder](https://jsonplaceholder.typicode.com/) API
# für unsere Beispiele.

# %% [markdown]
#
# ### Einen einzelnen Post abrufen
#
# - Protocol: HTTPS
# - Server: `jsonplaceholder.typicode.com`
# - Route: `/posts/1`
# - Full URL: `https://jsonplaceholder.typicode.com/posts/1`
#
# - REST-Methode: GET

# %%
# !pip install requests --root-user-action ignore

# %%
import requests

# %%
url = "https://jsonplaceholder.typicode.com/posts/1"

# %%
response = requests.get(url)

# %% [markdown]
#
# ### Das Response-Objekt

# %%
response

# %% [markdown]
#
# ### Status Code prüfen

# %%
response.status_code

# %% [markdown]
#
# ### Header ansehen

# %%
response.headers

# %% [markdown]
#
# ### Header-Details

# %%
type(response.headers)

# %%
dict(response.headers)

# %%
response.headers["Content-Type"]

# %% [markdown]
#
# ### Die Roh-Daten

# %%
response.content

# %% [markdown]
#
# ### Roh-Daten verarbeiten

# %%
type(response.content)

# %%
type(response.content.decode())

# %%
print(response.content.decode())

# %% [markdown]
#
# ### Als Python-Dictionary

# %%
response.json()

# %% [markdown]
#
# ### Mit den Daten arbeiten

# %%
post = response.json()

# %%
type(post)

# %%
post

# %%
post["title"]

# %%
post["body"]

# %% [markdown]
#
# ### Mehrere Posts abrufen

# %%
posts_url = "https://jsonplaceholder.typicode.com/posts"

# %%
posts_response = requests.get(posts_url)

# %%
posts_response.status_code

# %%
posts = posts_response.json()

# %%
type(posts)

# %%
len(posts)

# %%
posts

# %% [markdown]
#
# Zeige die ersten Posts an

# %%
for post in posts[:3]:
    print(f"{post['id']}: {post['title']}")

# %% [markdown]
#
# ## Fehlerbehandlung
#
# Was passiert, wenn etwas schief geht?

# %%
posts_response.status_code

# %%
missing = requests.get("https://jsonplaceholder.typicode.com/posts/999999")

# %%
missing.status_code

# %%
missing.content

# %% [markdown]
#
# ### Fehler automatisch erkennen
#
# - `response.raise_for_status()` wirft eine Exception bei Fehlern
# - Bei erfolgreichen Antworten passiert nichts

# %%
posts_response.raise_for_status()  # Kein Fehler

# %%
try:
    missing.raise_for_status()
except requests.HTTPError as e:
    print(f"Caught Error: {e}")


# %% [markdown]
#
# ### Best Practice: Immer Fehler prüfen

# %%
def get_post(post_id):
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"
    response = requests.get(url)
    response.raise_for_status()  # Throw error if bad response
    return response.json()


# %% [markdown]
#
# 1 ist eine gültige ID für Posts

# %%
get_post(1)

# %% [markdown]
#
# 999.999 ist keine gültige ID für Posts

# %%
try:
    get_post(999_999)
except requests.HTTPError as e:
    print(f"Post nicht gefunden: {e}")

# %% [markdown]
#
# ## Praktisches Beispiel: Länderinformationen
#
# Wir verwenden die [REST Countries API](https://restcountries.com/)

# %%
country_url = "https://restcountries.com/v3.1/name/"


# %% [markdown]
#
# ### Länderinformationen abrufen

# %%
country_response = requests.get(country_url + "Germany")


# %% [markdown]
#
# ### Ausdrucken der Informationen

# %%
def print_single_country_info(country):
    """Gibt Informationen über ein Land aus"""
    name = country["name"]["common"]
    capital = country.get("capital", ["N/A"])[0]
    population = country["population"]

    print(f"\nLand: {name}")
    print(f"Hauptstadt: {capital}")
    print(f"Bevölkerung: {population:,}")


# %%
def print_country_info(country_json):
    for country in country_json:
        print_single_country_info(country)


# %%
country_response.json()

# %% [markdown]
#
# ### Anfordern und Ausdrucken von Informationen

# %%
print_country_info(country_response.json())


# %% [markdown]
#
# ### Verbesserte Version mit Fehlerbehandlung

# %%
def get_country_info(country_name):
    response = requests.get(country_url + country_name)
    response.raise_for_status()
    return response.json()


# %%
def request_and_print_country_info(country_name):
    countries = get_country_info(country_name)
    print_country_info(countries)


# %% [markdown]
#
# ### Verbesserte Version testen

# %%
# Successful call
try:
    request_and_print_country_info("Germany")
except requests.HTTPError as e:
    print(f"Error occurred: {e}")

# %%
# Not found
try:
    get_country_info("Atlantis")
except requests.HTTPError as e:
    print(f"Error occurred: {e}")

# %% [markdown]
#
# ## POST Requests: Daten senden
#
# Mit POST können wir neue Daten an einen Server senden.

# %%
posts_url = "https://jsonplaceholder.typicode.com/posts"

# %%
new_post = {
    "userId": 1,
    "title": "Mein neuer Post",
    "body": "Das ist der Inhalt meines Posts.",
}

# %%
post_response = requests.post(posts_url, json=new_post)

# %%
post_response.status_code  # 201 = Created

# %% [markdown]
#
# ### Das erstellte Objekt

# %%
created_post = post_response.json()

# %%
created_post

# %% [markdown]
#
# ## PUT/PATCH: Daten aktualisieren

# %%
update_url = "https://jsonplaceholder.typicode.com/posts/1"
updated_data = {"title": "Aktualisierter Titel"}

# %%
patch_response = requests.patch(update_url, json=updated_data)

# %%
patch_response.status_code  # 200 = OK

# %%
patch_response.json()


# %% [markdown]
#
# ## Zusammenfassung
#
# **Das `requests` Package:**
# - `requests.get(url)` - Daten abrufen
# - `requests.post(url, json=...)` - Daten senden
# - `response.json()` - Daten als Python-Objekt
# - `response.raise_for_status()` - Fehler erkennen

# %% [markdown]
#
# ## Nächste Schritte
#
# Mit diesem Wissen können Sie:
# - Mit APIs von Web-Diensten arbeiten
# - Daten von KI-Services wie ChatGPT abrufen
# - Eigene Programme schreiben, die Web Services nutzen
# - REST APIs verstehen und verwenden

# %% [markdown]
#
# ## Mini-Workshop: Produkt-API
#
# Die Website [https://dummyjson.com/](https://dummyjson.com/) stellt Test-Daten
# bereit.
#
# Unter [https://dummyjson.com/products](https://dummyjson.com/products) können
# Sie Produkt-Informationen abrufen.
#
# Jedes Produkt hat diese Struktur:
#
# ```python
# {
#   "id": 1,
#   "title": "...",
#   "price": 123,
#   "brand": "...",
#   "category": "..."
# }
# ```
#
# Die Antwort hat die Form:
# ```python
# {"products": [{"id": 1, ...}, {"id": 2, ...}, ...]}
# ```

# %% [markdown]
#
# ### Aufgabe 1: Alle Produkte abrufen
#
# Schreiben Sie eine Funktion `get_products()`, die alle Produkte von
# `https://dummyjson.com/products` abruft und die Liste der Produkte zurückgibt.

# %%
product_url = "https://dummyjson.com/products"


# %%
def get_products():
    """Ruft alle Produkte ab"""
    response = requests.get(product_url)
    response.raise_for_status()
    return response.json()["products"]


# %%
products = get_products()

# %% [markdown]
#
# ### Produkte anzeigen

# %%
print(f"Anzahl der Produkte: {len(products)}")

# %%
# Zeige die ersten 3 Produkte
for product in products[:3]:
    print(f"{product['id']}: {product['title']} - ${product['price']}")


# %% [markdown]
#
# ### Aufgabe 2: Ein einzelnes Produkt abrufen
#
# Schreiben Sie eine Funktion `get_single_product(product_id)`, die ein
# einzelnes Produkt mit der angegebenen ID abruft.
#
# Hinweis: Die URL ist `https://dummyjson.com/products/{id}`

# %%
def get_single_product(product_id):
    """Ruft ein einzelnes Produkt ab"""
    url = f"{product_url}/{product_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


# %%
product = get_single_product(1)
product

# %% [markdown]
#
# ### Produktdetails anzeigen

# %%
print(f"Produkt: {product['title']}")
print(f"Preis: ${product['price']}")
print(f"Marke: {product['brand']}")
print(f"Kategorie: {product['category']}")


# %% [markdown]
#
# ### Aufgabe 3: Produkte suchen (Optional)
#
# Die API unterstützt auch Suche: `https://dummyjson.com/products/search?q=phone`
#
# Schreiben Sie eine Funktion `search_products(query)`, die nach Produkten
# sucht und die Ergebnisse zurückgibt.

# %%
def search_products(query):
    """Sucht nach Produkten"""
    url = f"{product_url}/search"
    response = requests.get(url, params={"q": query})
    response.raise_for_status()
    return response.json()["products"]


# %%
phone_products = search_products("phone")

# %% [markdown]
#
# ### Suchergebnisse anzeigen

# %%
print(f"Gefundene Produkte: {len(phone_products)}")
for product in phone_products[:5]:
    print(f"- {product['title']}")

# %%
