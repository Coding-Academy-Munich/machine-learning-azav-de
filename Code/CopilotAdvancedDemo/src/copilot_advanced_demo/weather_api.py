"""Weather API client for fetching weather data."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
from datetime import datetime


@dataclass
class WeatherData:
    """Weather data for a location."""

    city: str
    country: str
    temperature: float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city": self.city,
            "country": self.country,
            "temperature": self.temperature,
            "feels_like": self.feels_like,
            "humidity": self.humidity,
            "description": self.description,
            "wind_speed": self.wind_speed,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ForecastDay:
    """Forecast data for a single day."""

    date: datetime
    temp_min: float
    temp_max: float
    description: str
    precipitation_chance: float


class WeatherAPIClient:
    """Client for interacting with weather API."""

    def __init__(self, config):
        self.config = config
        self._cache: Dict[str, tuple] = {}

    def get_current_weather(self, city: str) -> WeatherData:
        """
        Get current weather for a city.

        Args:
            city: City name (e.g., "London" or "London,UK")

        Returns:
            WeatherData object with current conditions
        """
        # Check cache first
        cache_key = f"current:{city}"
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.config.cache_ttl:
                return data

        # In a real implementation, this would make an HTTP request
        # For demo purposes, return mock data
        weather = WeatherData(
            city=city.split(",")[0],
            country=city.split(",")[1] if "," in city else "Unknown",
            temperature=22.5,
            feels_like=23.0,
            humidity=65,
            description="Partly cloudy",
            wind_speed=12.5,
            timestamp=datetime.now(),
        )

        self._cache[cache_key] = (weather, datetime.now())
        return weather

    def get_forecast(self, city: str, days: int = 5) -> List[ForecastDay]:
        """
        Get weather forecast for a city.

        Args:
            city: City name
            days: Number of days to forecast (1-7)

        Returns:
            List of ForecastDay objects
        """
        # Mock implementation
        forecast = []
        base_date = datetime.now()

        for i in range(days):
            day = ForecastDay(
                date=datetime(base_date.year, base_date.month, base_date.day + i),
                temp_min=15.0 + i,
                temp_max=25.0 + i,
                description="Sunny" if i % 2 == 0 else "Cloudy",
                precipitation_chance=0.1 * i,
            )
            forecast.append(day)

        return forecast

    def clear_cache(self):
        """Clear the weather data cache."""
        self._cache.clear()
