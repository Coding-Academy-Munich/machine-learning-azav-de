"""Tests for dashboard functionality."""

import pytest
from copilot_advanced_demo.dashboard import WeatherDashboard


@pytest.fixture
def dashboard(config):
    """Create dashboard instance."""
    return WeatherDashboard(config)


class TestWeatherDashboard:
    """Test WeatherDashboard class."""

    def test_init(self, config):
        """Test dashboard initialization."""
        dashboard = WeatherDashboard(config)
        assert dashboard.config == config
        assert dashboard.client is not None
        assert dashboard.favorite_cities == []

    def test_add_favorite_city(self, dashboard):
        """Test adding favorite cities."""
        dashboard.add_favorite_city("London,UK")
        assert "London,UK" in dashboard.favorite_cities

        # Should not add duplicates
        dashboard.add_favorite_city("London,UK")
        assert dashboard.favorite_cities.count("London,UK") == 1

    def test_remove_favorite_city(self, dashboard):
        """Test removing favorite cities."""
        dashboard.add_favorite_city("London,UK")
        result = dashboard.remove_favorite_city("London,UK")

        assert result is True
        assert "London,UK" not in dashboard.favorite_cities

        # Try removing non-existent city
        result = dashboard.remove_favorite_city("Paris,FR")
        assert result is False

    def test_display_current_weather(self, dashboard):
        """Test displaying current weather."""
        result = dashboard.display_current_weather("London,UK")

        assert "London" in result
        assert "Temperature:" in result
        assert "°C" in result

    def test_display_forecast(self, dashboard):
        """Test displaying forecast."""
        result = dashboard.display_forecast("London,UK", 3)

        assert "Weather Forecast:" in result
        assert "°C" in result

    def test_display_favorites_summary_empty(self, dashboard):
        """Test favorites summary with no favorites."""
        result = dashboard.display_favorites_summary()
        assert "No favorite cities" in result

    def test_display_favorites_summary_with_cities(self, dashboard):
        """Test favorites summary with cities added."""
        dashboard.add_favorite_city("London,UK")
        dashboard.add_favorite_city("Paris,FR")

        result = dashboard.display_favorites_summary()
        assert "Favorite Cities Weather:" in result
        assert "London,UK" in result
        assert "Paris,FR" in result

    def test_get_weekly_analysis(self, dashboard):
        """Test weekly analysis."""
        result = dashboard.get_weekly_analysis("London,UK")

        assert "Weekly Analysis" in result
        assert "Average temperature:" in result
        assert "Warmest day:" in result
        assert "Coldest day:" in result

    def test_dashboard_main_function_exists(self):
        """Test that main() function exists and is callable."""
        from copilot_advanced_demo.dashboard import main
        assert callable(main)


class TestDashboardIntegration:
    """Integration tests for dashboard."""

    def test_full_workflow(self, dashboard):
        """Test complete dashboard workflow."""
        # Add favorites
        dashboard.add_favorite_city("London,UK")
        dashboard.add_favorite_city("Paris,FR")
        dashboard.add_favorite_city("Berlin,DE")

        assert len(dashboard.favorite_cities) == 3

        # Get current weather for each
        for city in dashboard.favorite_cities:
            weather = dashboard.display_current_weather(city)
            assert weather is not None
            assert len(weather) > 0

        # Get favorites summary
        summary = dashboard.display_favorites_summary()
        assert "London,UK" in summary
        assert "Paris,FR" in summary
        assert "Berlin,DE" in summary

        # Remove one city
        dashboard.remove_favorite_city("Paris,FR")
        assert len(dashboard.favorite_cities) == 2

        # Get analysis
        analysis = dashboard.get_weekly_analysis("London,UK")
        assert "Average temperature:" in analysis
