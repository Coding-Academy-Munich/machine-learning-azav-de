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

# %% [markdown]
#
# ### Das Response-Objekt

# %%

# %% [markdown]
#
# ### Status Code prüfen

# %%

# %% [markdown]
#
# ### Header ansehen

# %%

# %% [markdown]
#
# ### Header-Details

# %%

# %%

# %%

# %% [markdown]
#
# ### Die Roh-Daten

# %%

# %% [markdown]
#
# ### Roh-Daten verarbeiten

# %%

# %%

# %%

# %% [markdown]
#
# ### Als Python-Dictionary

# %%

# %% [markdown]
#
# ### Mit den Daten arbeiten

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ### Mehrere Posts abrufen

# %%
posts_url = "https://jsonplaceholder.typicode.com/posts"

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
#
# Zeige die ersten Posts an

# %%

# %% [markdown]
#
# ## Fehlerbehandlung
#
# Was passiert, wenn etwas schief geht?

# %%

# %%
missing = requests.get("https://jsonplaceholder.typicode.com/posts/999999")

# %%

# %%

# %% [markdown]
#
# ### Fehler automatisch erkennen
#
# - `response.raise_for_status()` wirft eine Exception bei Fehlern
# - Bei erfolgreichen Antworten passiert nichts

# %%

# %%

# %% [markdown]
#
# ### Best Practice: Immer Fehler prüfen

# %%

# %% [markdown]
#
# 1 ist eine gültige ID für Posts

# %%

# %% [markdown]
#
# 999.999 ist keine gültige ID für Posts

# %%

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

# %% [markdown]
#
# ### Anfordern und Ausdrucken von Informationen

# %%

# %% [markdown]
#
# ### Verbesserte Version mit Fehlerbehandlung

# %%

# %%

# %% [markdown]
#
# ### Verbesserte Version testen

# %%

# %%

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

# %%

# %% [markdown]
#
# ### Das erstellte Objekt

# %%

# %%

# %% [markdown]
#
# ## PUT/PATCH: Daten aktualisieren

# %%

# %%

# %%

# %%


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

# %%

# %% [markdown]
#
# ### Produkte anzeigen

# %%

# %%

# %% [markdown]
#
# ### Aufgabe 2: Ein einzelnes Produkt abrufen
#
# Schreiben Sie eine Funktion `get_single_product(product_id)`, die ein
# einzelnes Produkt mit der angegebenen ID abruft.
#
# Hinweis: Die URL ist `https://dummyjson.com/products/{id}`

# %%

# %%

# %% [markdown]
#
# ### Produktdetails anzeigen

# %%

# %% [markdown]
#
# ### Aufgabe 3: Produkte suchen (Optional)
#
# Die API unterstützt auch Suche: `https://dummyjson.com/products/search?q=phone`
#
# Schreiben Sie eine Funktion `search_products(query)`, die nach Produkten
# sucht und die Ergebnisse zurückgibt.

# %%

# %%

# %% [markdown]
#
# ### Suchergebnisse anzeigen

# %%

# %%
