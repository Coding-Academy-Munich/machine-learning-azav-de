"""Configuration settings for WeatherDash."""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Config:
    """Application configuration."""

    api_key: str
    base_url: str = "https://api.openweathermap.org/data/2.5"
    units: str = "metric"
    language: str = "en"
    cache_ttl: int = 300  # 5 minutes
    timeout: int = 10

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            raise ValueError("WEATHER_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            base_url=os.getenv("WEATHER_BASE_URL", cls.base_url),
            units=os.getenv("WEATHER_UNITS", cls.units),
            language=os.getenv("WEATHER_LANGUAGE", cls.language),
            cache_ttl=int(os.getenv("WEATHER_CACHE_TTL", cls.cache_ttl)),
            timeout=int(os.getenv("WEATHER_TIMEOUT", cls.timeout)),
        )


# Default configuration for development
DEFAULT_CONFIG = Config(
    api_key="demo_key",
    base_url="https://api.openweathermap.org/data/2.5",
)
