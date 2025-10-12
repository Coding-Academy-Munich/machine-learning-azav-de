# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Spezialisierte API Bibliotheken</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Spezialisierte API Bibliotheken
#
# - Viele Web Services bieten eigene Python-Bibliotheken an
# - Diese **vereinfachen** die Kommunikation mit dem Service
# - Sie kümmern sich um:
#   - Die korrekte Formatierung der Anfragen
#   - Die Verarbeitung der Antworten
#   - Fehlerbehandlung
#   - Authentifizierung
# - **Vorteil:** Sie müssen die API-Details nicht kennen

# %% [markdown]
#
# ## Beispiel: Wikipedia
#
# - Wikipedia bietet eine API für den Zugriff auf Daten
# - Die **MediaWiki API** ist komplex und umfangreich
# - Die `pymediawiki`-Bibliothek macht den Zugriff einfach
#
# **Installation:**
# ```bash
# pip install pymediawiki
# ```

# %% [markdown]
#
# ## Der direkte Weg: Mit `requests`
#
# Lassen Sie uns zunächst versuchen, Daten von Wikipedia mit `requests`
# abzurufen.

# %%
# !pip install requests --root-user-action ignore

# %%
import requests


# %% [markdown]
#
# ### Wikipedia-Suche mit `requests`

# %%
def search_wikipedia_raw(search_term):
    """Search Wikipedia articles with the MediaWiki API"""
    api_url = "https://en.wikipedia.org/w/api.php"

    # Parameters are complex!
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_term,
        "format": "json",
        "srlimit": 5,
    }

    # Add User-Agent header to comply with Wikipedia's robot policy
    headers = {"User-Agent": "PythonCourse/1.0 (Educational purposes)"}

    response = requests.get(api_url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


# %% [markdown]
#
# ### Suche nach 'Python'

# %%
result = search_wikipedia_raw("Python")

# %% [markdown]
#
# ### Das Ergebnis ist komplex verschachtelt

# %%
result

# %% [markdown]
#
# ### Die Daten extrahieren
#
# Wir müssen tief in die verschachtelte Struktur eintauchen

# %%
search_results = result.get("query", {}).get("search", [])

# %%
for item in search_results[:3]:
    print(f"- {item['title']}")


# %% [markdown]
#
# ### Einen Artikel-Zusammenfassung abrufen

# %%
def get_wikipedia_summary_raw(title):
    """Ruft die Zusammenfassung eines Wikipedia-Artikels ab"""
    api_url = "https://en.wikipedia.org/w/api.php"

    # Even more complex parameters!
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
    }

    # Add User-Agent header to comply with Wikipedia's robot policy
    headers = {"User-Agent": "PythonCourse/1.0 (Educational purposes)"}

    response = requests.get(api_url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


# %%
summary_result = get_wikipedia_summary_raw("Python (programming language)")

# %%
summary_result

# %% [markdown]
#
# ### Die Zusammenfassung extrahieren
#
# Die Struktur ist noch komplizierter!

# %%
pages = summary_result.get("query", {}).get("pages", {})

# %% [markdown]
#
# Die Seiten-ID ist dynamisch - wir müssen sie finden

# %%
page_id = list(pages.keys())[0]

# %% [markdown]
#
# Dann können wir damit in `pages` indizieren

# %%
extract = pages[page_id].get("extract", "")

# %%
print(extract[:300] + "...")

# %% [markdown]
#
# ## Das Problem mit dem direkten Ansatz
#
# - Die API-Parameter sind **komplex** und schwer zu merken
# - Die Antwort-Struktur ist **verschachtelt** und umständlich
# - Man muss die **API-Dokumentation** ständig konsultieren
# - **Fehleranfällig:** Kleine Fehler in Parametern führen zu Problemen
# - Viel **Boilerplate-Code** nötig

# %% [markdown]
#
# ## Der einfache Weg: Die `pymediawiki` Bibliothek
#
# Die `pymediawiki`-Bibliothek vereinfacht alles!
#
# ```bash
# pip install pymediawiki
# ```

# %%
# !pip install pymediawiki --root-user-action ignore

# %%
from mediawiki import MediaWiki

# %% [markdown]
#
# ### MediaWiki-Objekt erstellen

# %%
api_url = "https://en.wikipedia.org/w/api.php"

# %%
mediawiki = MediaWiki(api_url, user_agent="PythonCourse")

# %% [markdown]
#
# ### Suchen - viel einfacher!
#
# Einfach eine Liste von Titeln

# %%
page_titles = mediawiki.search("Python")

# %%
page_titles

# %%
for title in page_titles[:5]:
    print(f"- {title}")

# %% [markdown]
#
# ### Zusammenfassung abrufen - noch einfacher!
#
# Direkt der Text, keine verschachtelte Struktur!

# %%
summary = mediawiki.summary("Python (programming language)", sentences=3)

# %%
print(summary)

# %% [markdown]
#
# ## Vergleich: Direkt vs. Bibliothek
#
# **Mit `requests` (direkt):**
# ```python
# # Viele Parameter, komplexe Struktur
# result = requests.get(api_url, params={
#     "action": "query",
#     "list": "search",
#     "srsearch": search_term,
#     "format": "json",
#     "srlimit": 5,
# })
# titles = result["query"]["search"]  # Verschachtelt!
# ```
#
# **Mit `pymediawiki`:**
# ```python
# # Einfach und klar
# titles = mediawiki.search("Python")
# ```

# %% [markdown]
#
# ### Vollständigen Seiteninhalt abrufen

# %%
page = mediawiki.page("Python (programming language)")

# %% [markdown]
#
# Verschieden Informationen sind abrufbar:

# %%
page.title

# %%
page.url

# %%
len(page.links)

# %% [markdown]
#
# Die ersten 4 Links

# %%
page.links[:4]

# %% [markdown]
#
# Die ersten 4 Kategorien

# %%
page.categories[:4]

# %% [markdown]
#
# Die ersten 4 Abschnitte

# %%
page.sections[:4]

# %% [markdown]
#
# ## Andere Beispiele für Custom API Libraries
#
# Viele beliebte Services bieten eigene Libraries:
#
# - **OpenAI:** `openai` - für ChatGPT, DALL-E, etc.
# - **Google:** `google-cloud-*` - für Google Cloud Services
# - **AWS:** `boto3` - für Amazon Web Services
# - **Twitter:** `tweepy` - für Twitter API
# - **Stripe:** `stripe` - für Zahlungsabwicklung
# - **GitHub:** `PyGithub` - für GitHub API
# - **Spotify:** `spotipy` - für Spotify API

# %% [markdown]
#
# ## Wann sollten Sie Custom Libraries verwenden?
#
# **Vorteile:**
# - ✅ **Einfacher zu benutzen** - weniger Code, klare API
# - ✅ **Weniger Fehler** - Library kümmert sich um Details
# - ✅ **Bessere Dokumentation** - meist gut dokumentiert
# - ✅ **Wartung** - Updates werden von den Entwicklern gemacht
#
# **Nachteile:**
# - ❌ **Zusätzliche Abhängigkeit** - ein weiteres Package
# - ❌ **Lernkurve** - Sie müssen die Library lernen
# - ❌ **Mögliche Limitierungen** - Library bietet vielleicht nicht alles

# %% [markdown]
#
# ## Zusammenfassung
#
# ### Custom API Libraries
# - Vereinfachen die Arbeit mit komplexen APIs
# - Kümmern sich um technische Details
# - Bieten eine klare, pythonische Schnittstelle
#
# ### Empfehlung
# - Prüfen Sie immer, ob eine offizielle Library existiert
# - Bei LLM-Services: Nutzen Sie die offiziellen SDKs (z.B. `openai`)

# %% [markdown]
#
# ### `pymediawiki` Beispiel
#
# - `mediawiki.search(term)` - Suche nach Artikeln
# - `mediawiki.summary(title)` - Zusammenfassung abrufen
# - `mediawiki.page(title)` - Vollständige Seite mit allen Details


# %% [markdown]
#
# ## Mini-Workshop: Wikipedia Explorer
#
# Erstellen Sie ein Programm, das:
# 1. Nach einem Begriff sucht
# 2. Alle gefundenen Artikel auflistet
# 3. Für den ersten Artikel:
#    - Die Zusammenfassung ausgibt
#    - Die ersten 5 Links auflistet
#    - Die ersten 5 Kategorien auflistet

# %%
from textwrap import wrap


# %%
def create_mediawiki_instance():
    """Creates and returns a MediaWiki instance"""
    return MediaWiki("https://en.wikipedia.org/w/api.php", user_agent="WikiExplorer")


# %%
def search_and_display_articles(mediawiki, search_term: str):
    """Searches for articles and displays them. Returns list of titles or None."""
    print(f"Searching for: '{search_term}'\n")
    page_titles = mediawiki.search(search_term)

    if not page_titles:
        print("No articles found.")
        return None

    print(f"Found articles ({len(page_titles)}):")
    for i, title in enumerate(page_titles, 1):
        print(f"  {i}. {title}")

    return page_titles


# %%
def print_article_header(title: str):
    """Prints a formatted header for an article"""
    print(f"\n{'='*70}")
    print(f"Details for: {title}")
    print("=" * 70)


# %%
def display_article_summary(mediawiki, title: str, sentences: int = 2):
    """Retrieves and displays the summary of an article"""
    summary = mediawiki.summary(title, sentences=sentences)
    print(f"\nSummary:")
    print("\n".join(wrap(summary, width=70)))


# %%
def display_article_links(page, num_links: int = 5):
    """Displays the first N links from an article"""
    print(f"\nFirst {num_links} links:")
    for link in page.links[:num_links]:
        print(f"  - {link}")


# %%
def display_article_categories(page, num_categories: int = 5):
    """Displays the first N categories from an article"""
    print(f"\nFirst {num_categories} categories:")
    for category in page.categories[:num_categories]:
        print(f"  - {category}")


# %%
def display_article_details(mediawiki, title: str):
    """Displays detailed information about an article"""
    print_article_header(title)
    display_article_summary(mediawiki, title)

    # Get full page for links and categories
    page = mediawiki.page(title)
    display_article_links(page)
    display_article_categories(page)


# %%
def explore_wikipedia(search_term: str):
    """Explores a Wikipedia topic"""
    mediawiki = create_mediawiki_instance()

    # Search for articles and display them
    page_titles = search_and_display_articles(mediawiki, search_term)

    if not page_titles:
        return

    # Display details for the first article
    first_title = page_titles[0]
    display_article_details(mediawiki, first_title)


# %%
explore_wikipedia("Artificial Intelligence")

# %%
explore_wikipedia("Climate Change")

# %%
