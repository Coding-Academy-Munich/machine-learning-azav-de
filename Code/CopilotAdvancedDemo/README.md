# WeatherDash - Copilot Advanced Demo

A modern Python weather dashboard application demonstrating best practices and GitHub Copilot's advanced features.

## Features

- Current weather display
- Multi-day forecasts
- Favorite cities management
- Weather data analysis
- Command-line interface

## Installation

Install with pip:
```bash
pip install -e .
```

Or with uv (faster):
```bash
uv pip install -e .
```

Install development dependencies:
```bash
pip install -e ".[dev]"
# or
uv pip install -e ".[dev]"
```

## Project Structure

```
copilot-advanced-demo/
  src/
    copilot_advanced_demo/
      __init__.py       # Package exports
      config.py         # Configuration settings
      weather_api.py    # API client and data models
      data_processor.py # Data transformation
      dashboard.py      # Main dashboard logic
      cli.py           # Command-line interface
  tests/
    test_weather_api.py
  pyproject.toml       # Modern Python project configuration
  .github/
    copilot-instructions.md  # Custom Copilot instructions
```

## Usage

### Command Line

After installation, use the `weather-dash` command:

```bash
# Current weather
weather-dash "London,UK"

# 3-day forecast
weather-dash "Paris,FR" --forecast --days 3

# Weekly analysis
weather-dash "Berlin,DE" --analysis
```

Or run as a Python module:

```bash
# Using the CLI module
python -m copilot_advanced_demo.cli "Munich,DE" -f

# Run the dashboard demo
python -m copilot_advanced_demo.dashboard
```

### Python API
```python
from copilot_advanced_demo import WeatherDashboard, Config

config = Config.from_env()  # or Config(api_key="your_key")
dashboard = WeatherDashboard(config)

print(dashboard.display_current_weather("London,UK"))
```

## Testing

```bash
pytest tests/
```

## Usage with Copilot

This project demonstrates:

1. **@workspace** - Ask about the entire codebase
2. **Copilot Edits** - Multi-file refactoring  
3. **Custom Instructions** - Project-specific guidance
4. **Agent Mode** - Implementing new features
5. **Modern Python packaging** - pyproject.toml, src layout
