"""Tests for CLI functionality."""

import pytest
from unittest.mock import patch, MagicMock
import sys

from copilot_advanced_demo.cli import main
from copilot_advanced_demo.config import Config


class TestCLI:
    """Test CLI entry point."""

    @patch('copilot_advanced_demo.cli.WeatherDashboard')
    @patch('copilot_advanced_demo.cli.Config')
    def test_current_weather(self, mock_config, mock_dashboard):
        """Test current weather display via CLI."""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config.from_env.side_effect = ValueError("No API key")
        mock_config.return_value = mock_config_instance
        
        mock_dash_instance = MagicMock()
        mock_dash_instance.display_current_weather.return_value = "Weather data"
        mock_dashboard.return_value = mock_dash_instance
        
        # Test CLI
        with patch.object(sys, 'argv', ['weather-dash', 'London,UK']):
            result = main()
        
        assert result == 0
        mock_dashboard.assert_called_once()
        mock_dash_instance.display_current_weather.assert_called_once_with('London,UK')

    @patch('copilot_advanced_demo.cli.WeatherDashboard')
    @patch('copilot_advanced_demo.cli.Config')
    def test_forecast(self, mock_config, mock_dashboard):
        """Test forecast display via CLI."""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        mock_dash_instance = MagicMock()
        mock_dash_instance.display_forecast.return_value = "Forecast data"
        mock_dashboard.return_value = mock_dash_instance
        
        with patch.object(sys, 'argv', ['weather-dash', 'Paris,FR', '--forecast', '--days', '3']):
            result = main()
        
        assert result == 0
        mock_dash_instance.display_forecast.assert_called_once_with('Paris,FR', 3)

    @patch('copilot_advanced_demo.cli.WeatherDashboard')
    @patch('copilot_advanced_demo.cli.Config')
    def test_weekly_analysis(self, mock_config, mock_dashboard):
        """Test weekly analysis via CLI."""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        mock_dash_instance = MagicMock()
        mock_dash_instance.get_weekly_analysis.return_value = "Analysis data"
        mock_dashboard.return_value = mock_dash_instance
        
        with patch.object(sys, 'argv', ['weather-dash', 'Berlin,DE', '--analysis']):
            result = main()
        
        assert result == 0
        mock_dash_instance.get_weekly_analysis.assert_called_once_with('Berlin,DE')

    @patch('copilot_advanced_demo.cli.WeatherDashboard')
    @patch('copilot_advanced_demo.cli.Config')
    def test_with_api_key(self, mock_config, mock_dashboard):
        """Test CLI with API key argument."""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        mock_dash_instance = MagicMock()
        mock_dash_instance.display_current_weather.return_value = "Weather data"
        mock_dashboard.return_value = mock_dash_instance
        
        with patch.object(sys, 'argv', ['weather-dash', 'London,UK', '--api-key', 'test_key']):
            result = main()
        
        assert result == 0
        mock_config.assert_called_once_with(api_key='test_key')

    @patch('copilot_advanced_demo.cli.WeatherDashboard')
    @patch('copilot_advanced_demo.cli.Config')
    def test_error_handling(self, mock_config, mock_dashboard):
        """Test CLI error handling."""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        mock_dash_instance = MagicMock()
        mock_dash_instance.display_current_weather.side_effect = Exception("API Error")
        mock_dashboard.return_value = mock_dash_instance
        
        with patch.object(sys, 'argv', ['weather-dash', 'London,UK']):
            result = main()
        
        assert result == 1
