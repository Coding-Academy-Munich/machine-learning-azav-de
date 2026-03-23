"""Data processing utilities for weather data."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from .weather_api import WeatherData, ForecastDay


def calculate_average_temperature(forecast: List[ForecastDay]) -> float:
    """Calculate average temperature from forecast data."""
    if not forecast:
        return 0.0

    total = sum((day.temp_min + day.temp_max) / 2 for day in forecast)
    return total / len(forecast)


def find_warmest_day(forecast: List[ForecastDay]) -> Optional[ForecastDay]:
    """Find the warmest day in the forecast."""
    if not forecast:
        return None

    return max(forecast, key=lambda d: d.temp_max)


def find_coldest_day(forecast: List[ForecastDay]) -> Optional[ForecastDay]:
    """Find the coldest day in the forecast."""
    if not forecast:
        return None

    return min(forecast, key=lambda d: d.temp_min)


def format_temperature(temp: float, units: str = "metric") -> str:
    """Format temperature with appropriate unit symbol."""
    if units == "metric":
        return f"{temp:.1f}°C"
    elif units == "imperial":
        return f"{temp:.1f}°F"
    else:
        return f"{temp:.1f}K"


def format_weather_summary(weather: WeatherData) -> str:
    """Create a human-readable weather summary."""
    return (
        f"Weather in {weather.city}, {weather.country}:\n"
        f"  Temperature: {format_temperature(weather.temperature)}\n"
        f"  Feels like: {format_temperature(weather.feels_like)}\n"
        f"  Humidity: {weather.humidity}%\n"
        f"  Conditions: {weather.description}\n"
        f"  Wind: {weather.wind_speed} m/s"
    )


def format_forecast_summary(forecast: List[ForecastDay]) -> str:
    """Create a human-readable forecast summary."""
    if not forecast:
        return "No forecast data available."

    lines = ["Weather Forecast:"]
    for day in forecast:
        date_str = day.date.strftime("%A, %B %d")
        lines.append(
            f"  {date_str}: {format_temperature(day.temp_min)} - "
            f"{format_temperature(day.temp_max)}, {day.description}"
        )

    return "\n".join(lines)


def filter_rainy_days(forecast: List[ForecastDay], threshold: float = 0.5) -> List[ForecastDay]:
    """Filter forecast to show only days with high precipitation chance."""
    return [day for day in forecast if day.precipitation_chance >= threshold]


def group_by_condition(forecast: List[ForecastDay]) -> Dict[str, List[ForecastDay]]:
    """Group forecast days by weather condition."""
    groups = {}
    for day in forecast:
        condition = day.description
        if condition not in groups:
            groups[condition] = []
        groups[condition].append(day)
    return groups
