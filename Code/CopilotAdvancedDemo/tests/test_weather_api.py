"""Tests for weather API client."""

import pytest
from datetime import datetime

from copilot_advanced_demo.weather_api import WeatherAPIClient, WeatherData, ForecastDay
from copilot_advanced_demo.config import Config


@pytest.fixture
def config():
    return Config(api_key="test_key")


@pytest.fixture
def client(config):
    return WeatherAPIClient(config)


class TestWeatherAPIClient:
    def test_get_current_weather(self, client):
        weather = client.get_current_weather("London,UK")

        assert weather.city == "London"
        assert weather.country == "UK"
        assert isinstance(weather.temperature, float)
        assert isinstance(weather.humidity, int)

    def test_get_forecast(self, client):
        forecast = client.get_forecast("London,UK", days=5)

        assert len(forecast) == 5
        assert all(isinstance(day, ForecastDay) for day in forecast)

    def test_cache_works(self, client):
        # First call
        weather1 = client.get_current_weather("London,UK")
        # Second call should return cached data
        weather2 = client.get_current_weather("London,UK")

        assert weather1.timestamp == weather2.timestamp

    def test_clear_cache(self, client):
        client.get_current_weather("London,UK")
        assert len(client._cache) > 0

        client.clear_cache()
        assert len(client._cache) == 0


class TestWeatherData:
    def test_to_dict(self):
        weather = WeatherData(
            city="London",
            country="UK",
            temperature=20.0,
            feels_like=21.0,
            humidity=65,
            description="Sunny",
            wind_speed=10.0,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
        )

        data = weather.to_dict()

        assert data["city"] == "London"
        assert data["temperature"] == 20.0
        assert "timestamp" in data


# TODO: Add more tests for error handling
# TODO: Add tests for different units (metric, imperial)
# TODO: Add integration tests with mock HTTP responses
