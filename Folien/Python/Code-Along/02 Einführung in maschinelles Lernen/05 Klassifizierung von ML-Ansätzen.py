# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Klassifizierung von ML-Ansätzen</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Klassifizierung von ML-Ansätzen
#
# Es gibt verschiedene Möglichkeiten, ML-Ansätze zu klassifizieren:
#
# - Nach der verwendeten Technik
# - Nach der Art des Problems
# - Nach der Art des Lernens
# - Nach der Ebene der Arbeit am ML-Stack

# %% [markdown]
#
# ## ML-Techniken
#
# - **Klassisches ML**
#   - Traditionelle Algorithmen (z.B. Lineare Regression, Decision Trees, SVM)
#   - Meist für strukturierte Daten
#   - Interpretierbar und gut verstanden
# - **Deep Learning / Neuronale Netze**
#   - Mehrschichtige neuronale Netze
#   - Besonders gut für unstrukturierte Daten (Bilder, Text, Audio)
#   - Benötigt oft große Datenmengen
# - **Ensemble-Methoden**
#   - Kombination mehrerer Modelle (z.B. Random Forests, Gradient Boosting)
#   - Oft sehr gute Performance

# %% [markdown]
#
# ## Probleme, die ML lösen kann
#
# - **Klassifikation**
#   - Zuordnung von Eingaben zu Kategorien
#   - Beispiel: Spam-Erkennung, Bilderkennung
# - **Regression**
#   - Vorhersage von kontinuierlichen Werten
#   - Beispiel: Preisvorhersage, Wettervorhersage
# - **Clustering**
#   - Gruppierung ähnlicher Datenpunkte
#   - Beispiel: Kundensegmentierung

# %% [markdown]
#
# ## Weitere ML-Probleme
#
# - **Generierung**
#   - Erzeugung neuer Daten ähnlich den Trainingsdaten
#   - Beispiel: Textgenerierung, Bildsynthese, Musikkomposition
# - **Dimensionsreduktion**
#   - Reduzierung der Anzahl der Merkmale
#   - Beispiel: Visualisierung hochdimensionaler Daten
# - **Anomalieerkennung**
#   - Identifikation ungewöhnlicher Datenpunkte
#   - Beispiel: Betrugserkennung, Fehlerdiagnose

# %% [markdown]
#
# ## Ansätze zum Lernen
#
# - **Überwachtes Lernen (Supervised Learning)**
#   - Modell lernt aus gelabelten Daten
#   - Eingabe-Ausgabe-Paare sind bekannt
#   - Beispiel: Bilderkennung mit gelabelten Bildern
# - **Unüberwachtes Lernen (Unsupervised Learning)**
#   - Modell lernt Muster ohne Labels
#   - Nur Eingabedaten sind verfügbar
#   - Beispiel: Clustering, Dimensionsreduktion

# %% [markdown]
#
# ## Weitere Lernansätze
#
# - **Semi-Supervised Learning**
#   - Kombination aus gelabelten und ungelabelten Daten
#   - Wenige Labels, viele unlabelte Daten
#   - Beispiel: Webseiten-Klassifikation
# - **Reinforcement Learning (Verstärkendes Lernen)**
#   - Modell lernt durch Interaktion mit einer Umgebung
#   - Belohnungen für gute Aktionen, Bestrafungen für schlechte
#   - Beispiel: Spielende KI, Robotersteuerung

# %% [markdown]
#
# ## Ebenen der ML-Arbeit
#
# Je nach Anwendungsfall und Expertise können wir auf verschiedenen Ebenen mit
# ML arbeiten:
#
# - Entwicklung neuer Modellarchitekturen
# - Training von Modellen
# - Arbeit mit vortrainierten Modellen
# - Finetuning vortrainierter Modelle
# - Aufbau von Systemen, die Modelle enthalten
# - Verwendung von LLMs für Programmierung/andere Aufgaben

# %% [markdown]
#
# ## Entwicklung von Modellarchitekturen
#
# - **Höchste Komplexität**
# - Erfordert tiefes Verständnis von:
#   - Mathematik (lineare Algebra, Statistik, Optimierung)
#   - ML-Theorie
#   - Programmierung
# - Beispiele:
#   - Entwicklung neuer Netzwerkarchitekturen
#   - Forschung an neuen Lernalgorithmen
# - Typischerweise in der Forschung oder bei großen Tech-Unternehmen

# %% [markdown]
#
# ## Training von Modellen
#
# - Training eines Modells von Grund auf
# - Erfordert:
#   - Große Datenmengen
#   - Rechenressourcen
#   - Verständnis von Hyperparameter-Tuning
# - Beispiele:
#   - Training eines Bildklassifikators auf eigenen Daten
#   - Training eines Sprachmodells
# - Oft zeitaufwendig und kostspielig

# %% [markdown]
#
# ## Arbeit mit vortrainierten Modellen
#
# - Verwendung eines bereits trainierten Modells
# - Keine oder minimale Anpassung erforderlich
# - Beispiele:
#   - Verwendung von GPT-5 über API
#   - Verwendung von YOLO für Objekterkennung
#   - Verwendung von BERT für Textklassifikation
# - **Vorteil**: Schneller Einstieg, geringe Kosten
# - **Nachteil**: Weniger Flexibilität

# %% [markdown]
#
# ## Finetuning vortrainierter Modelle
#
# - Anpassung eines vortrainierten Modells an eine spezifische Aufgabe
# - Erfordert:
#   - Weniger Daten als Training von Grund auf
#   - Verständnis von Transfer Learning
#   - Moderate Rechenressourcen
# - Beispiele:
#   - Anpassung eines Sprachmodells an eine bestimmte Domäne
#   - Anpassung eines Bildklassifikators an eigene Kategorien
# - **Balance**: Gute Performance mit moderatem Aufwand

# %% [markdown]
#
# ## Aufbau von Systemen mit ML-Modellen
#
# - Integration von ML-Modellen in größere Systeme
# - Erfordert:
#   - Software-Engineering-Kenntnisse
#   - Verständnis von ML-Operations (MLOps)
#   - Systemarchitektur-Kenntnisse
# - Beispiele:
#   - Empfehlungssystem für E-Commerce
#   - Chatbot mit Sprachverständnis
#   - Automatisierte Qualitätskontrolle
# - **Fokus**: Produktivierung und Wartung

# %% [markdown]
#
# ## Verwendung von LLMs als Werkzeug
#
# - Nutzung von Large Language Models für alltägliche Aufgaben
# - Keine ML-Kenntnisse erforderlich
# - Beispiele:
#   - Code-Generierung mit GitHub Copilot
#   - Texterstellung mit ChatGPT
#   - Datenanalyse mit LLM-Unterstützung
# - **Einstiegshürde**: Sehr niedrig
# - **Wichtig**: Verständnis der Limitationen und ethischen Aspekte

# %% [markdown]
#
# ## Übersicht: Ebenen der ML-Arbeit
#
# | Ebene                 | Komplexität  | Ressourcen   | Flexibilität |
# |-----------------------|--------------|--------------|--------------|
# | Modellarchitekturen   | Sehr hoch    | Sehr hoch    | Sehr hoch    |
# | Training              | Hoch         | Hoch         | Hoch         |
# | Finetuning            | Mittel       | Mittel       | Mittel       |
# | Vortrainierte Modelle | Niedrig      | Niedrig      | Niedrig      |
# | Systembau             | Mittel       | Mittel       | Hoch         |
# | LLM-Nutzung           | Sehr niedrig | Sehr niedrig | Niedrig      |

# %% [markdown]
#
# ## Welche Ebene ist die richtige?
#
# Die Wahl der Ebene hängt ab von:
#
# - **Verfügbaren Ressourcen**
#   - Zeit, Budget, Rechenleistung
# - **Expertise**
#   - ML-Kenntnisse, Programmierkenntnisse
# - **Anforderungen**
#   - Genauigkeit, Anpassbarkeit, Geschwindigkeit
# - **Daten**
#   - Menge und Qualität der verfügbaren Daten

# %% [markdown]
#
# ## Empfehlungen für Einsteiger
#
# - **Beginnen Sie mit LLM-Nutzung**
#   - Verstehen Sie die Möglichkeiten und Grenzen
# - **Arbeiten Sie mit vortrainierten Modellen**
#   - Lernen Sie, wie man sie effektiv einsetzt
# - **Experimentieren Sie mit Finetuning**
#   - Wenn Sie spezifische Anforderungen haben
# - **Lernen Sie die Grundlagen**
#   - Verstehen Sie, wie ML funktioniert
# - **Bauen Sie schrittweise Expertise auf**
#   - Von einfachen zu komplexeren Aufgaben

# %% [markdown]
#
# ## Zusammenfassung
#
# - ML kann nach verschiedenen Kriterien klassifiziert werden
# - Verschiedene Techniken für verschiedene Probleme
# - Unterschiedliche Lernansätze je nach verfügbaren Daten
# - Verschiedene Ebenen der ML-Arbeit je nach Expertise und Anforderungen
# - Wählen Sie die Ebene, die zu Ihren Ressourcen und Zielen passt
