# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in Web Services</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Was sind Web Services?
#
# - Programme, die über das Internet miteinander kommunizieren
# - Ein Programm (der **Client**) fragt Daten an
# - Ein anderes Programm (der **Server**) antwortet mit den Daten
# - Ähnlich wie ein Restaurantbesuch:
#   - Sie (Client) bestellen beim Kellner
#   - Die Küche (Server) bereitet das Essen zu
#   - Der Kellner bringt Ihnen das Essen

# %% [markdown]
#
# ## Beispiele für Web Services
#
# - **Wetter-Apps:** Fragen Wetterdaten von einem Server ab
# - **Social Media:** Laden Posts, Bilder, Kommentare
# - **Online-Shopping:** Produktinformationen, Preise, Verfügbarkeit
# - **Navigationsdienste:** Google Maps fragt Kartendaten ab
# - **KI-Services:** ChatGPT, DALL-E, Übersetzungsdienste

# %% [markdown]
#
# ## Wie funktioniert das Internet? (Vereinfacht)
#
# - Jeder Computer im Internet hat eine **Adresse** (wie eine Postadresse)
# - Diese Adresse heißt **URL** (Uniform Resource Locator)
# - Beispiel: `https://www.example.com/users/123`
#   - `https://` - Das Protokoll (wie die Sprache)
#   - `www.example.com` - Die Serveradresse
#   - `/users/123` - Der Pfad (die Route) zur Ressource (welche Daten Sie wollen)

# %% [markdown]
#
# ## HTTP: Die Sprache des Internets
#
# - **HTTP** = HyperText Transfer Protocol
# - Das "Protokoll" für die Kommunikation zwischen Client und Server
# - Wie ein Formular mit verschiedenen Feldern:
#   - **Methode:** Was möchten Sie tun?
#   - **URL:** Wo sind die Daten?
#   - **Header:** Zusätzliche Informationen
#   - **Body:** Daten, die Sie senden (optional)

# %% [markdown]
#
# ## HTTP-Methoden: Was möchten Sie tun?
#
# - **GET:** Daten abrufen (wie ein Buch aus der Bibliothek ausleihen)
#   - "Gib mir die Benutzerdaten"
#   - "Zeig mir die Produktliste"
# - **POST:** Neue Daten erstellen (wie einen Brief abschicken)
#   - "Erstelle einen neuen Benutzer"
#   - "Sende diese Nachricht"
# - **PUT/PATCH:** Daten aktualisieren (wie einen Text korrigieren)
# - **DELETE:** Daten löschen (wie etwas in den Papierkorb werfen)

# %% [markdown]
#
# ## HTTP Status Codes: Hat es funktioniert?
#
# Der Server antwortet immer mit einem **Status Code:**
#
# - **200 OK:** Alles gut! Hier sind Ihre Daten
# - **201 Created:** Erfolgreich erstellt
# - **400 Bad Request:** Ihre Anfrage war fehlerhaft
# - **401 Unauthorized:** Sie müssen sich anmelden
# - **404 Not Found:** Diese Daten existieren nicht
# - **500 Internal Server Error:** Problem auf dem Server

# %% [markdown]
#
# ## Was ist eine API?
#
# - **API** = Application Programming Interface
# - Eine "Schnittstelle" für Programme, um miteinander zu kommunizieren
# - Ähnlich wie das Bestellen in einem Restaurant:
#   - Sie müssen nicht wissen, wie die Küche funktioniert
#   - Sie sagen dem Kellner einfach, welches Gericht Sie möchten
#     - "Ich hätte gerne die Spaghetti Bolognese."
#     - "Nummer 32 bitte."
#   - Der Kellner bringt Ihnen das Essen
# - Anders als in einem Restaurant müssen Sie sich bei einer API an eine strikte
#   Spezifikation halten

# %% [markdown]
#
# ## APIs in der Praxis
#
# - **Web APIs:** Ermöglichen Zugriff auf Web-Dienste
# - Definieren:
#   - Welche Daten verfügbar sind
#   - Wie Sie diese Daten anfordern
#   - Welches Format die Antwort hat
# - Beispiel: Die OpenAI API ermöglicht Zugriff auf ChatGPT

# %% [markdown]
#
# ## REST APIs: Ein beliebtes Design-Muster
#
# - **REST** = Representational State Transfer
# - Ein Stil für das Design von Web Services
# - Hauptmerkmale:
#   - API zur Verwaltung von "Ressourcen" (Daten wie Benutzer, Produkte, etc.)
#   - Jede Ressource hat eine URL
#   - Abrufen/Erzeugen/Löschen/... von Ressourcen mit Standard-HTTP-Methoden
#   - Daten meist im JSON-Format
#   - Zustandslos (jede Anfrage ist unabhängig)

# %% [markdown]
#
# ## JSON: Das Datenformat
#
# - **JSON** = JavaScript Object Notation
# - Das am häufigsten verwendete Format für Web Services
# - Ähnlich wie Python-Dictionaries und -Listen
# - Beispiel:
#
# ```json
# {
#   "name": "Max Mustermann",
#   "age": 30,
#   "hobbies": ["Lesen", "Wandern"]
# }
# ```
