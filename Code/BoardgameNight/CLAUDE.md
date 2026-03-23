# CLAUDE.md - AI Assistant Instructions

This file provides guidance for AI coding assistants (Claude Code, GitHub Copilot) when working on this project.

## Project Overview

**Board Game Night Organizer** - A Python application to manage board game collections, track play sessions, and provide smart game recommendations.

### Tech Stack

- **Python 3.12+**
- **Package Manager:** uv
- **Testing:** pytest with pytest-cov
- **Linting/Formatting:** ruff
- **Type Checking:** mypy (strict mode)
- **Pre-commit Hooks:** pre-commit

## Essential Commands

### Run After Every Change

```bash
# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check --fix .

# Type checking
uv run mypy src/

# Run tests
uv run pytest
```

### Quick Validation (All Checks)

```bash
uv run ruff format . && uv run ruff check --fix . && uv run mypy src/ && uv run pytest
```

### Install Dependencies

```bash
uv sync --extra dev
```

### Run Pre-commit on All Files

```bash
uv run pre-commit run --all-files
```

## Project Structure

```text
src/boardgame_night/
├── __init__.py          # Package init with version
├── py.typed             # PEP 561 type hint marker
├── models/              # Domain models (Game, Player, PlaySession)
├── services/            # Business logic (recommendations, statistics)
└── storage/             # Data persistence (JSON, future: SQLite)

tests/
├── conftest.py          # Shared fixtures
├── test_models/         # Unit tests for models
├── test_services/       # Unit tests for services
└── test_integration/    # Integration tests
```

## Coding Conventions

### Type Hints

- **Always use type hints** for function parameters and return types
- Use `from __future__ import annotations` for forward references
- Prefer `list[T]` over `List[T]` (Python 3.9+ syntax)
- Use `| None` instead of `Optional[T]`

```python
def find_games(
    min_players: int,
    max_playtime: int | None = None,
) -> list[Game]:
    ...
```

### Dataclasses

- Use `@dataclass` for domain models
- Use `frozen=True` for immutable value objects
- Use `slots=True` for memory efficiency

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Game:
    name: str
    min_players: int
    max_players: int
```

### Error Handling

- Create specific exception classes in `exceptions.py`
- Raise exceptions with descriptive messages
- Document exceptions in docstrings

### Testing

- Follow TDD: write tests first, then implement
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Include edge cases and error conditions
- Use fixtures from `conftest.py` for common test data

```python
def test_find_games_with_player_count_filters_correctly():
    ...

def test_find_games_with_zero_players_raises_error():
    ...
```

## Domain Model

### Core Entities

1. **Game** - A board game with metadata
   - name, min_players, max_players, min_playtime, max_playtime
   - complexity (1-5 scale), categories, mechanics

2. **Player** - A person who plays games
   - name, preferences, play history

3. **PlaySession** - A record of a game being played
   - game, players, date, winner, duration, notes

4. **Collection** - A user's game collection
   - games, owner

### Business Rules

- A game cannot have min_players > max_players
- A game cannot have min_playtime > max_playtime
- Complexity must be between 1.0 and 5.0
- A play session must have at least min_players for the game
- A play session cannot exceed max_players for the game

## Feature Implementation Order

When implementing features, follow this order:

1. **Domain Models** - Start with Game, Player, PlaySession dataclasses
2. **Validation** - Add business rule validation
3. **Collection Management** - CRUD operations for games
4. **Play Session Logging** - Record and query play sessions
5. **Recommendations** - Filter and suggest games
6. **Statistics** - Play history analysis

## Common Pitfalls

- Don't modify test assertions to make them pass - fix the code
- Don't skip type hints - mypy is in strict mode
- Don't use `Any` type unless absolutely necessary
- Don't forget to run all checks before committing

## Commit Message Format

```text
<type>: <short description>

<optional body>

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
