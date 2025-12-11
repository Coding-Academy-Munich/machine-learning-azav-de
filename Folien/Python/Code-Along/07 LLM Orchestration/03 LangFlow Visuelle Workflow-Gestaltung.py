# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LangFlow: Visuelle Workflow-Gestaltung</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Von Code zu Visual Design
#
# - Bisher: Workflows in Code schreiben (LangChain, LangGraph)
# - **LangFlow**: Workflows **visuell** designen
# - Drag & Drop Interface
# - Kein Code nötig (aber möglich!)
#
# Ideal für:
# - Schnelles Prototyping
# - Nicht-Programmierer im Team
# - Komplexe Workflows visualisieren

# %% [markdown]
#
# ## Was ist LangFlow?
#
# - **Open Source** visuelles Tool für LangChain-Workflows
# - Web-basiertes Interface (läuft lokal)
# - **Komponenten** per Drag & Drop verbinden
# - Export zu Python-Code möglich
# - **Integration** mit dem LangChain-Ökosystem
#
# Entwickelt von der LangChain-Community

# %% [markdown]
#
# ## Installation und Start
#
# ```bash
# pip install langflow
# langflow run
# ```
#
# Öffnet automatisch den Browser mit LangFlow UI (http://localhost:7860)
#
# **Alternativ mit Docker:**
# ```bash
# docker run -p 7860:7860 langflowai/langflow
# ```

# %% [markdown]
#
# ## Die LangFlow Oberfläche
#
# <!-- Placeholder: Screenshot of LangFlow UI overview -->
# <img src="img/langflow_ui_overview.png" alt="LangFlow UI: Komponenten-Panel links, Canvas in der Mitte, Eigenschaften rechts" />
#
# **Hauptbereiche:**
# - **Links**: Komponenten-Panel (Inputs, Models, Outputs, etc.)
# - **Mitte**: Canvas zum Verbinden der Komponenten
# - **Rechts**: Eigenschaften der ausgewählten Komponente
# - **Oben**: Toolbar (Speichern, Ausführen, Teilen)

# %% [markdown]
#
# ## Komponenten-Kategorien
#
# | Kategorie | Beschreibung | Beispiele |
# |-----------|--------------|-----------|
# | **Inputs** | Benutzereingaben | Chat Input, Text Input |
# | **Models** | LLM-Verbindungen | OpenAI, Anthropic, Ollama |
# | **Outputs** | Ergebnisanzeige | Chat Output, Text Output |
# | **Prompts** | Prompt Templates | Prompt Template |
# | **Data** | Datenquellen | File Loader, URL Loader |
# | **Vector Stores** | Vektordatenbanken | Chroma, Pinecone |
# | **Embeddings** | Embedding-Modelle | OpenAI Embeddings |
# | **Chains** | Vorgefertigte Chains | Conversation Chain |

# %% [markdown]
#
# ## Chatbot bauen: Schritt 1 - Chat Input
#
# <!-- Placeholder: Screenshot showing Chat Input component being added -->
# <img src="img/langflow_step1_chat_input.png" alt="Chat Input Komponente auf dem Canvas" />
#
# 1. Im Komponenten-Panel: **Inputs** → **Chat Input**
# 2. Per Drag & Drop auf den Canvas ziehen
# 3. Der Chat Input nimmt Benutzernachrichten entgegen

# %% [markdown]
#
# ## Chatbot bauen: Schritt 2 - LLM Model
#
# <!-- Placeholder: Screenshot showing OpenAI model component -->
# <img src="img/langflow_step2_model.png" alt="OpenAI Model Komponente mit Konfiguration" />
#
# 1. **Models** → **OpenAI** (oder anderer Anbieter)
# 2. Auf Canvas ziehen
# 3. **API Key** in den Eigenschaften eingeben
# 4. **Model Name** auswählen (z.B. `gpt-4o-mini`)
#
# **Für OpenRouter**: Base URL auf `https://openrouter.ai/api/v1` setzen

# %% [markdown]
#
# ## Chatbot bauen: Schritt 3 - Chat Output
#
# <!-- Placeholder: Screenshot showing Chat Output component -->
# <img src="img/langflow_step3_chat_output.png" alt="Chat Output Komponente" />
#
# 1. **Outputs** → **Chat Output**
# 2. Auf Canvas ziehen
# 3. Zeigt die LLM-Antworten an

# %% [markdown]
#
# ## Chatbot bauen: Schritt 4 - Verbinden
#
# <!-- Placeholder: Screenshot showing connected components -->
# <img src="img/langflow_step4_connected.png" alt="Alle Komponenten verbunden: Input → Model → Output" />
#
# 1. **Chat Input** Ausgang → **OpenAI** Eingang verbinden
# 2. **OpenAI** Ausgang → **Chat Output** Eingang verbinden
# 3. Fertig! Der Datenfluss ist: Input → Model → Output

# %% [markdown]
#
# ## Chatbot bauen: Schritt 5 - Testen
#
# <!-- Placeholder: Screenshot showing the Playground -->
# <img src="img/langflow_step5_playground.png" alt="LangFlow Playground mit Chat-Interface" />
#
# 1. Klick auf **Playground** (unten rechts)
# 2. Chat-Interface öffnet sich
# 3. Nachricht eingeben und testen!

# %% [markdown]
#
# ## RAG-Flow bauen: Übersicht
#
# Für ein RAG-System brauchen wir mehr Komponenten:
#
# ```
# Dokument Loader → Text Splitter → Embeddings → Vector Store
#                                                    ↓
# Chat Input → Retriever → Prompt Template → LLM → Chat Output
# ```

# %% [markdown]
#
# ## RAG-Flow: Dokumente laden
#
# <!-- Placeholder: Screenshot showing document loading components -->
# <img src="img/langflow_rag_documents.png" alt="File Loader und Text Splitter Komponenten" />
#
# 1. **Data** → **File Loader**: Lädt PDF, TXT, etc.
# 2. **Processing** → **Text Splitter**: Teilt in Chunks
# 3. Verbinden: File Loader → Text Splitter

# %% [markdown]
#
# ## RAG-Flow: Vector Store
#
# <!-- Placeholder: Screenshot showing vector store setup -->
# <img src="img/langflow_rag_vectorstore.png" alt="Chroma Vector Store mit Embeddings" />
#
# 1. **Embeddings** → **OpenAI Embeddings**
# 2. **Vector Stores** → **Chroma** (oder Pinecone, etc.)
# 3. Verbinden: Text Splitter → Chroma, Embeddings → Chroma

# %% [markdown]
#
# ## RAG-Flow: Retrieval Chain
#
# <!-- Placeholder: Screenshot showing complete RAG flow -->
# <img src="img/langflow_rag_complete.png" alt="Vollständiger RAG-Flow mit allen Komponenten" />
#
# 1. **Retriever** aus Vector Store erstellen
# 2. **Prompt Template** für RAG-Kontext
# 3. Alles verbinden: Input → Retriever → Prompt → LLM → Output

# %% [markdown]
#
# ## Export zu Python
#
# LangFlow kann Flows als Python-Code exportieren:
#
# 1. **Menü** → **Export** → **Python Code**
# 2. Code wird generiert
# 3. Kann in eigenen Projekten verwendet werden
#
# **Wichtig**: Exportierter Code kann als Basis für weitere Anpassungen dienen

# %% [markdown]
#
# ## Beispiel: Exportierter Code
#
# ```python
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
#
# llm = ChatOpenAI(model="gpt-4o-mini", api_key="...")
#
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant."),
#     ("human", "{input}")
# ])
#
# chain = prompt | llm
# result = chain.invoke({"input": "Hello!"})
# ```

# %% [markdown]
#
# ## LangFlow vs. Code
#
# | Aspekt | LangFlow | Code (LangChain/LangGraph) |
# |--------|----------|---------------------------|
# | **Lernkurve** | Niedrig | Mittel-Hoch |
# | **Flexibilität** | Begrenzt | Sehr hoch |
# | **Prototyping** | Sehr schnell | Langsamer |
# | **Versionierung** | Schwieriger | Git-freundlich |
# | **Debugging** | Visuell | Detaillierter |
# | **Team-Arbeit** | Nicht-Entwickler können helfen | Entwickler-fokussiert |

# %% [markdown]
#
# ## Wann LangFlow nutzen?
#
# **Gut geeignet für:**
# - Schnelles Prototyping neuer Ideen
# - Demonstrationen für Stakeholder
# - Einfache Chatbots und Q&A-Systeme
# - Teams mit Nicht-Programmierern
#
# **Besser Code nutzen für:**
# - Komplexe Geschäftslogik
# - Production-Systeme
# - Feinabstimmung und Optimierung
# - Versionskontrolle und CI/CD

# %% [markdown]
#
# ## Zusammenfassung
#
# - **LangFlow**: Visuelle Alternative zu Code
# - **Drag & Drop** Interface für Workflows
# - **Komponenten**: Inputs, Models, Outputs, Vector Stores, etc.
# - **Export**: Flows können als Python-Code exportiert werden
# - **Use Case**: Prototyping und einfache Anwendungen
#
# **Empfehlung**: LangFlow für Prototypen, dann zu Code migrieren

# %% [markdown]
#
# ## Workshop: LangFlow erkunden
#
# **Aufgabe**: Erstellen Sie in LangFlow:
#
# 1. Einen einfachen Chatbot mit System-Prompt
# 2. Testen Sie verschiedene Modelle
# 3. Fügen Sie einen Prompt Template hinzu
# 4. Exportieren Sie den Flow als Python-Code
#
# **Bonus**: Bauen Sie einen einfachen RAG-Flow mit einer Textdatei

# %% [markdown]
#
# ## Workshop-Hilfe: LangFlow starten
#
# ```bash
# # Terminal öffnen
# pip install langflow
# langflow run
# ```
#
# Browser öffnet sich automatisch!

# %%
