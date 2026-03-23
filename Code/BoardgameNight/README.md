# Board Game Night Organizer

Organize board game nights: manage collections, track play sessions, and get smart game recommendations.

## Features

- **Collection Management**: Track your board game collection with details like player counts, playtime, and complexity
- **Play Session Logging**: Record who played, when, and who won
- **Smart Recommendations**: Get game suggestions based on player count, available time, and preferences
- **Player Profiles**: Track player preferences and gaming history

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/hoelzl/boardgame-night.git
cd boardgame-night

# Install dependencies (including dev dependencies)
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

## Development

### Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=boardgame_night --cov-report=term-missing
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code (with auto-fix)
uv run ruff check --fix .

# Type checking
uv run mypy src/
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
uv run pre-commit run --all-files
```

## Project Structure

```text
boardgame-night/
├── src/
│   └── boardgame_night/
│       ├── __init__.py
│       └── py.typed           # PEP 561 marker for type hints
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Shared pytest fixtures
│   └── test_*.py              # Test files
├── pyproject.toml             # Project configuration
├── CLAUDE.md                  # AI assistant instructions
└── README.md
```

## License

MIT
