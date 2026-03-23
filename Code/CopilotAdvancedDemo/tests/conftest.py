"""Shared pytest fixtures for tests."""

import pytest
from copilot_advanced_demo.config import Config


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(api_key="test_key")
