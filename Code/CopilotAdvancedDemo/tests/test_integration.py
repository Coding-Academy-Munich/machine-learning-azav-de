"""Integration tests for entry points and package structure."""

import subprocess
import sys
import pytest


class TestEntryPoints:
    """Test that all entry points work correctly."""

    def test_weather_dash_command_exists(self):
        """Test that weather-dash command is available."""
        result = subprocess.run(
            ["weather-dash", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Weather Dashboard" in result.stdout
        assert "City name" in result.stdout

    def test_weather_dash_command_runs(self):
        """Test that weather-dash command runs successfully."""
        result = subprocess.run(
            ["weather-dash", "London,UK"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Weather in London" in result.stdout
        assert "Temperature:" in result.stdout

    def test_cli_module_runnable(self):
        """Test that CLI module can be run directly."""
        result = subprocess.run(
            [sys.executable, "-m", "copilot_advanced_demo.cli", "Paris,FR"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Weather in Paris" in result.stdout

    def test_dashboard_module_runnable(self):
        """Test that dashboard module can be run directly."""
        result = subprocess.run(
            [sys.executable, "-m", "copilot_advanced_demo.dashboard"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Weather in London" in result.stdout
        assert "Favorite Cities" in result.stdout

    def test_package_importable(self):
        """Test that package can be imported."""
        result = subprocess.run(
            [sys.executable, "-c", "import copilot_advanced_demo; print('OK')"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_package_exports(self):
        """Test that package exports expected symbols."""
        result = subprocess.run(
            [
                sys.executable, 
                "-c",
                "from copilot_advanced_demo import Config, WeatherDashboard, WeatherAPIClient; print('OK')"
            ],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_old_src_structure_does_not_exist(self):
        """Test that old src/ files are removed (no duplicates)."""
        import os
        
        old_files = [
            "src/config.py",
            "src/dashboard.py",
            "src/data_processor.py",
            "src/weather_api.py",
        ]
        
        for old_file in old_files:
            assert not os.path.exists(old_file), f"Old file {old_file} should not exist"

    def test_new_package_structure_exists(self):
        """Test that new package structure exists."""
        import os
        
        new_files = [
            "src/copilot_advanced_demo/__init__.py",
            "src/copilot_advanced_demo/config.py",
            "src/copilot_advanced_demo/dashboard.py",
            "src/copilot_advanced_demo/data_processor.py",
            "src/copilot_advanced_demo/weather_api.py",
            "src/copilot_advanced_demo/cli.py",
        ]
        
        for new_file in new_files:
            assert os.path.exists(new_file), f"New file {new_file} should exist"
