# Copilot Instructions for WeatherDash

## Project Overview
WeatherDash is a Python weather dashboard application that fetches and displays weather data.

## Code Style Guidelines
- Use type hints for all function parameters and return values
- Follow PEP 8 naming conventions
- Use dataclasses for data structures
- Write docstrings for all public functions and classes
- Prefer composition over inheritance

## Architecture
- `src/config.py` - Configuration management
- `src/weather_api.py` - API client and data models
- `src/data_processor.py` - Data transformation utilities
- `src/dashboard.py` - Main dashboard logic

## Testing
- Use pytest for all tests
- Place tests in the `tests/` directory
- Use fixtures for common setup
- Aim for high test coverage on data processing functions

## Error Handling
- Always validate input parameters
- Use custom exceptions for domain-specific errors
- Log errors with appropriate context
- Never expose API keys in error messages

## When generating code
- Always add type hints
- Include docstrings with Args, Returns, and Raises sections
- Handle edge cases (empty lists, None values)
- Consider caching for API calls
