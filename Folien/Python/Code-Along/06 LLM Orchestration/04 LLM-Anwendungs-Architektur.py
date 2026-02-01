# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>LLM-Anwendungs-Architektur</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# ## Von Prototyp zu Produktion
#
# - Bisher: Funktionale Prototypen
# - **Jetzt**: Production-Ready Code
# - Was bedeutet das?
#   - Fehlerbehandlung
#   - Logging
#   - Code-Organisation
#   - Sicherheit

# %% [markdown]
#
# ## Architektur-Prinzipien
#
# 1. **Separation of Concerns**: UI, Logik, Daten trennen
# 2. **Error Handling**: Alle möglichen Fehler behandeln
# 3. **Logging**: Nachvollziehbarkeit
# 4. **Caching**: Kosten sparen
# 5. **Security**: API-Keys sicher, Input validieren

# %%
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% [markdown]
#
# ## Zusammenfassung
#
# - Production-ready Code ist mehr als funktional
# - Architektur, Sicherheit, Monitoring wichtig
# - Caching spart Kosten
# - Proper Error Handling essentiell
#
# **Nächster Schritt**: Workshop - Alles zusammen!

# %%
