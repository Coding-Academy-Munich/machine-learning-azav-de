"""Copilot Advanced Demo - A Python weather dashboard application."""

__version__ = "0.1.0"

from .config import Config, DEFAULT_CONFIG
from .weather_api import WeatherAPIClient, WeatherData, ForecastDay
from .dashboard import WeatherDashboard
from . import data_processor

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "WeatherAPIClient",
    "WeatherData",
    "ForecastDay",
    "WeatherDashboard",
    "data_processor",
]
