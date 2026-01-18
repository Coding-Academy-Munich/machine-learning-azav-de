# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Prompt Templates und LangSmith</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Das Problem: Starre Prompts
#
# Unser Chatbot kann nur **eine** Sache:
#
# ```python
# system_prompt = "Du bist ein Python-Tutor."
# ```
#
# Was wenn der Benutzer...
# - ...einen **Mathe**-Tutor möchte?
# - ...einen **Koch** als Assistent?
# - ...einen **Geschichtenerzähler**?

# %% [markdown]
#
# ## Naive Lösung: String-Formatierung
#
# ```python
# topic = "Kochen"
# system_prompt = f"Du bist ein {topic}-Experte."
# ```
#
# **Probleme**:
# - Kompliziert bei vielen Variablen
# - Fehleranfällig (vergessene `f`, falsche Variablen)
# - Schwer zu debuggen

# %% [markdown]
#
# ## Die Lösung: ChatPromptTemplate
#
# LangChain bietet ein **Template-System** für Prompts:
#
# - Variablen mit `{name}` definieren
# - Klar strukturiert und wiederverwendbar
# - Einfach zu testen und debuggen

# %%
import os
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# %%
model = "mistralai/ministral-14b-2512"

# %%
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model,
)

# %% [markdown]
#
# ## Erstes Template
#
# Ein Template für einen Experten zu beliebigem Thema:

# %%
template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du bist ein hilfreicher {topic}-Experte."),
        ("human", "{question}"),
    ]
)

# %% [markdown]
#
# Jetzt können wir das Template mit verschiedenen Werten füllen:

# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Verschiedene Themen, gleicher Code
#
# Das gleiche Template funktioniert für jedes Thema:

# %%

# %%

# %% [markdown]
#
# ## Aber Moment... was wurde wirklich gesendet?
#
# Wir haben `{topic}` durch "Kochen" ersetzt.
#
# **Aber wie wissen wir, dass es funktioniert hat?**
#
# - Was wenn ein Fehler passiert?
# - Wie debuggen wir komplexe Prompts?
# - Wie sehen wir, was das LLM wirklich bekommen hat?

# %% [markdown]
#
# ## LangSmith: Observability für LLM-Apps
#
# **LangSmith** ist ein Tool von LangChain für:
#
# - **Traces**: Jeden LLM-Aufruf sehen
# - **Debugging**: Fehler schnell finden
# - **Monitoring**: Performance überwachen
#
# **Kostenlos** für Entwicklung!

# %% [markdown]
#
# ## LangSmith einrichten
#
# 1. Account erstellen: https://smith.langchain.com
# 2. API-Key in `.env` speichern:
#    ```
#    LANGSMITH_API_KEY=lsv2_pt_...
#    ```
# 3. Tracing aktivieren:

# %%
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "ml-azav-course"
if os.getenv("LANGSMITH_API_KEY") is None:
    print("Warning: LANGSMITH_API_KEY not set in environment variables.")

# %% [markdown]
#
# ## Jetzt mit Tracing!
#
# Nach der Aktivierung wird jeder Aufruf in LangSmith sichtbar:

# %%

# %%

# %% [markdown]
#
# ## Was wir in LangSmith sehen
#
# Öffnen Sie https://smith.langchain.com und sehen Sie:
#
# - **Input**: Die formatierten Nachrichten
# - **Output**: Die LLM-Antwort
# - **Latenz**: Wie lange der Aufruf dauerte
# - **Token-Verbrauch**: Kosten pro Aufruf
#
# Sie können sehen, dass `{topic}` durch "Geschichte" ersetzt wurde!

# %% [markdown]
#
# ## Mehrere Variablen
#
# Templates können beliebig viele Variablen haben:

# %%
tutor_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Du bist ein {topic}-Tutor.
Dein Unterrichtsstil ist: {style}
Antworte auf {language}.""",
        ),
        ("human", "{question}"),
    ]
)

# %%

# %%

# %%

# %% [markdown]
#
# ## MessagesPlaceholder für Konversationen
#
# Für Chat-Anwendungen brauchen wir den **Konversationsverlauf**:

# %%
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# %%
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du bist ein hilfreicher {topic}-Experte."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# %% [markdown]
#
# ## Template mit History verwenden

# %%
history = [
    HumanMessage(content="Was ist Python?"),
    AIMessage(content="Python ist eine Programmiersprache..."),
]


# %%

# %%

# %%

# %%

# %% [markdown]
#
# ## Template-basierter Chatbot
#
# Jetzt kombinieren wir alles zu einem flexiblen Chatbot:

# %%
class TemplateChatbot:
    """A chatbot using prompt templates."""

    def __init__(self, topic, style="freundlich und hilfsbereit"):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model,
        )
        self.topic = topic
        self.style = style
        self.history = []

        self.template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Du bist ein {topic}-Experte. Dein Stil ist: {style}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    def chat(self, user_message):
        """Send a message and get a response."""
        messages = self.template.invoke(
            {
                "topic": self.topic,
                "style": self.style,
                "history": self.history,
                "input": user_message,
            }
        )

        response = self.llm.invoke(messages)

        self.history.append(HumanMessage(content=user_message))
        self.history.append(response)

        return response.content


# %% [markdown]
#
# ## Den flexiblen Chatbot testen

# %%

# %%

# %%

# %% [markdown]
#
# ## LangSmith zeigt alles
#
# In LangSmith können Sie jetzt sehen:
#
# 1. Wie das Template gefüllt wurde
# 2. Die History-Nachrichten
# 3. Die vollständige Anfrage an das LLM
# 4. Die Antwort und Token-Verbrauch
#
# **Debugging wird damit viel einfacher!**

# %% [markdown]
#
# ## Zusammenfassung
#
# - **ChatPromptTemplate**: Strukturierte, wiederverwendbare Prompts
# - **Variablen**: `{name}` für dynamische Werte
# - **MessagesPlaceholder**: Für Konversationsverlauf
# - **LangSmith**: Observability für alle LLM-Aufrufe
#   - Traces zeigen jeden Schritt
#   - Debugging wird einfach
#   - Kosten überwachen

# %% [markdown]
#
# ## Workshop: "Lerne Alles" Tutor
#
# **Ziel**: Ein Chatbot, der jedes Thema erklären kann!
#
# **Aufgaben**:
# 1. Template mit `{topic}` und `{style}` Variablen
# 2. Gradio Interface mit Eingabefeldern für Thema und Stil
# 3. LangSmith Traces überprüfen
# 4. **Bonus**: Verschiedene Lehrstile als Dropdown

# %%
import gradio as gr

# %% [markdown]
#
# ### Teil 1: Template und Chatbot

# %%
TEACHING_STYLES_DE = {
    "Sokratisch": "Du stellst Fragen, um den Lernenden zum Nachdenken zu bringen.",
    "Schritt-für-Schritt": "Du erklärst alles in kleinen, klaren Schritten.",
    "Mit Beispielen": "Du verwendest viele praktische Beispiele.",
    "Einfach": "Du erklärst alles so einfach wie möglich.",
}

TEACHING_STYLES_EN = {
    "Socratic": "You ask questions to make the learner think.",
    "Step-by-step": "You explain everything in small, clear steps.",
    "With examples": "You use many practical examples.",
    "Simple": "You explain everything as simply as possible.",
}

# %%
TEACHING_STYLES = TEACHING_STYLES_DE


# %%
def create_tutor_response(message, history, topic, style):
    """Tutor chatbot with configurable topic and style."""
    # TODO: Create template and generate response
    pass

# %% [markdown]
#
# ### Teil 2: Gradio Interface

# %%
# TODO: Create ChatInterface with topic and style inputs

# %%

# %% [markdown]
#
# ### Überprüfen Sie LangSmith!
#
# Nach dem Testen des Tutors:
#
# 1. Öffnen Sie https://smith.langchain.com
# 2. Wählen Sie das Projekt "ml-azav-course"
# 3. Sehen Sie sich die Traces an:
#    - Wie wurde `{topic}` ersetzt?
#    - Welcher Lehrstil wurde verwendet?
#    - Wie viele Token wurden verbraucht?

# %% [markdown]
#
# ## Workshop-Aufgaben Zusammenfassung
#
# ### Basis (Pflicht):
# 1. Template mit `{topic}` und `{style}` erstellen
# 2. Gradio Interface mit Eingabefeldern
# 3. LangSmith Traces überprüfen
#
# ### Erweitert (Optional):
# 4. Eigene Lehrstile hinzufügen
# 5. Token-Kosten pro Konversation anzeigen
# 6. Export der Konversation als Text

# %%
