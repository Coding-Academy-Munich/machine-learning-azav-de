"""Dashboard display functionality."""

from typing import List, Optional
from .weather_api import WeatherData, ForecastDay, WeatherAPIClient
from .data_processor import (
    format_weather_summary,
    format_forecast_summary,
    calculate_average_temperature,
    find_warmest_day,
    find_coldest_day,
)
from .config import Config


class WeatherDashboard:
    """Main dashboard class for displaying weather information."""

    def __init__(self, config: Config):
        self.config = config
        self.client = WeatherAPIClient(config)
        self.favorite_cities: List[str] = []

    def add_favorite_city(self, city: str) -> None:
        """Add a city to favorites."""
        if city not in self.favorite_cities:
            self.favorite_cities.append(city)

    def remove_favorite_city(self, city: str) -> bool:
        """Remove a city from favorites."""
        if city in self.favorite_cities:
            self.favorite_cities.remove(city)
            return True
        return False

    def display_current_weather(self, city: str) -> str:
        """Display current weather for a city."""
        weather = self.client.get_current_weather(city)
        return format_weather_summary(weather)

    def display_forecast(self, city: str, days: int = 5) -> str:
        """Display weather forecast for a city."""
        forecast = self.client.get_forecast(city, days)
        return format_forecast_summary(forecast)

    def display_favorites_summary(self) -> str:
        """Display weather summary for all favorite cities."""
        if not self.favorite_cities:
            return "No favorite cities added yet."

        summaries = []
        for city in self.favorite_cities:
            weather = self.client.get_current_weather(city)
            summaries.append(f"{city}: {weather.temperature}°C, {weather.description}")

        return "Favorite Cities Weather:\n" + "\n".join(f"  - {s}" for s in summaries)

    def get_weekly_analysis(self, city: str) -> str:
        """Get analysis of the weekly forecast."""
        forecast = self.client.get_forecast(city, 7)

        avg_temp = calculate_average_temperature(forecast)
        warmest = find_warmest_day(forecast)
        coldest = find_coldest_day(forecast)

        analysis = [
            f"Weekly Analysis for {city}:",
            f"  Average temperature: {avg_temp:.1f}°C",
        ]

        if warmest:
            analysis.append(
                f"  Warmest day: {warmest.date.strftime('%A')} ({warmest.temp_max:.1f}°C)"
            )
        if coldest:
            analysis.append(
                f"  Coldest day: {coldest.date.strftime('%A')} ({coldest.temp_min:.1f}°C)"
            )

        return "\n".join(analysis)


def main():
    """Main entry point for the dashboard."""
    from .config import DEFAULT_CONFIG

    dashboard = WeatherDashboard(DEFAULT_CONFIG)

    # Demo usage
    print(dashboard.display_current_weather("London,UK"))
    print()
    print(dashboard.display_forecast("London,UK", 3))
    print()

    dashboard.add_favorite_city("London,UK")
    dashboard.add_favorite_city("Paris,FR")
    dashboard.add_favorite_city("Berlin,DE")
    print(dashboard.display_favorites_summary())


if __name__ == "__main__":
    main()
