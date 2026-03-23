# GitHub Copilot Instructions

## Project Context

This is a Python 3.12+ project using:
- **uv** for package management
- **ruff** for linting and formatting
- **mypy** (strict mode) for type checking
- **pytest** for testing

## Code Quality Requirements

After making changes, run these commands:

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy src/
uv run pytest
```

## Coding Standards

1. **Always use type hints** - mypy runs in strict mode
2. **Use dataclasses** for domain models with `frozen=True, slots=True`
3. **Follow TDD** - write tests first, then implement
4. **Use descriptive test names** - `test_<function>_<scenario>_<expected>`

## Domain Model

- **Game**: Board game with player counts, playtime, complexity
- **Player**: Person with preferences and history
- **PlaySession**: Record of a game being played
- **Collection**: A user's game collection

## Business Rules

- min_players <= max_players
- min_playtime <= max_playtime
- complexity between 1.0 and 5.0
- play sessions must respect player count limits
