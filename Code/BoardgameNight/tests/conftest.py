"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_game_data() -> dict[str, object]:
    """Sample game data for testing."""
    return {
        "name": "Catan",
        "min_players": 3,
        "max_players": 4,
        "min_playtime": 60,
        "max_playtime": 120,
        "complexity": 2.3,
    }
