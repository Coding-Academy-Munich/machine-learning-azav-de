# Feature Implementation Guide

This document provides detailed specifications for implementing features in the Board Game Night Organizer. Each feature is described with enough detail that an AI coding assistant can implement it with minimal manual intervention.

## Implementation Philosophy

1. **Test-Driven Development (TDD)**: Write tests first, then implement
2. **Small Commits**: Commit after each successful step
3. **Type Safety**: All code must pass mypy strict mode
4. **Validation**: Enforce business rules at construction time

---

## Feature 1: Game Model

### Overview

Create a `Game` dataclass representing a board game with its metadata.

### Specification

**File:** `src/boardgame_night/models/game.py`

**Class:** `Game`

**Attributes:**
| Attribute | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `name` | `str` | Game title | Non-empty |
| `min_players` | `int` | Minimum player count | >= 1 |
| `max_players` | `int` | Maximum player count | >= min_players |
| `min_playtime` | `int` | Minimum playtime in minutes | >= 1 |
| `max_playtime` | `int` | Maximum playtime in minutes | >= min_playtime |
| `complexity` | `float` | Complexity rating | 1.0 to 5.0 |
| `categories` | `frozenset[str]` | Game categories | Optional, default empty |
| `mechanics` | `frozenset[str]` | Game mechanics | Optional, default empty |

**Validation Rules:**
1. `name` must be non-empty after stripping whitespace
2. `min_players` must be at least 1
3. `max_players` must be >= `min_players`
4. `min_playtime` must be at least 1
5. `max_playtime` must be >= `min_playtime`
6. `complexity` must be between 1.0 and 5.0 (inclusive)

**Methods:**
- `supports_player_count(count: int) -> bool`: Returns True if the game supports the given player count
- `fits_time_slot(minutes: int) -> bool`: Returns True if the game can be played within the given time

### Test Cases

**File:** `tests/test_models/test_game.py`

```python
# Test: Create valid game
def test_create_game_with_valid_data():
    game = Game(
        name="Catan",
        min_players=3,
        max_players=4,
        min_playtime=60,
        max_playtime=120,
        complexity=2.3,
    )
    assert game.name == "Catan"
    assert game.min_players == 3
    assert game.max_players == 4

# Test: Validation - empty name raises ValueError
def test_create_game_with_empty_name_raises_error():
    with pytest.raises(ValueError, match="name"):
        Game(name="", min_players=2, max_players=4, ...)

# Test: Validation - min_players > max_players raises ValueError
def test_create_game_with_invalid_player_range_raises_error():
    with pytest.raises(ValueError, match="player"):
        Game(name="Test", min_players=5, max_players=2, ...)

# Test: Validation - complexity out of range raises ValueError
def test_create_game_with_invalid_complexity_raises_error():
    with pytest.raises(ValueError, match="complexity"):
        Game(name="Test", ..., complexity=6.0)

# Test: supports_player_count returns True for valid count
def test_supports_player_count_within_range():
    game = Game(name="Test", min_players=2, max_players=4, ...)
    assert game.supports_player_count(3) is True

# Test: supports_player_count returns False for invalid count
def test_supports_player_count_outside_range():
    game = Game(name="Test", min_players=2, max_players=4, ...)
    assert game.supports_player_count(5) is False

# Test: fits_time_slot returns True when time is sufficient
def test_fits_time_slot_with_sufficient_time():
    game = Game(..., min_playtime=30, max_playtime=60)
    assert game.fits_time_slot(45) is True

# Test: fits_time_slot returns False when time is insufficient
def test_fits_time_slot_with_insufficient_time():
    game = Game(..., min_playtime=60, max_playtime=120)
    assert game.fits_time_slot(30) is False
```

### Implementation Steps

1. Create `src/boardgame_night/models/__init__.py`
2. Create `tests/test_models/__init__.py`
3. Write the test file `tests/test_models/test_game.py`
4. Run tests to confirm they fail
5. Implement `src/boardgame_night/models/game.py`
6. Run tests to confirm they pass
7. Run `uv run ruff format . && uv run ruff check --fix . && uv run mypy src/`
8. Commit: `git commit -m "feat: Add Game model with validation"`

---

## Feature 2: Player Model

### Overview

Create a `Player` dataclass representing a person who plays games.

### Specification

**File:** `src/boardgame_night/models/player.py`

**Class:** `Player`

**Attributes:**
| Attribute | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `name` | `str` | Player name | Non-empty |
| `id` | `str` | Unique identifier | Auto-generated UUID if not provided |
| `favorite_categories` | `frozenset[str]` | Preferred categories | Optional |
| `complexity_preference` | `tuple[float, float]` | Min/max complexity | Both 1.0-5.0, min <= max |

**Validation Rules:**
1. `name` must be non-empty after stripping whitespace
2. `complexity_preference` values must be between 1.0 and 5.0
3. `complexity_preference[0]` must be <= `complexity_preference[1]`

### Test Cases

**File:** `tests/test_models/test_player.py`

```python
# Test: Create valid player
def test_create_player_with_valid_data():
    player = Player(name="Alice")
    assert player.name == "Alice"
    assert player.id is not None  # Auto-generated

# Test: Create player with custom ID
def test_create_player_with_custom_id():
    player = Player(name="Bob", id="bob-123")
    assert player.id == "bob-123"

# Test: Empty name raises ValueError
def test_create_player_with_empty_name_raises_error():
    with pytest.raises(ValueError, match="name"):
        Player(name="")

# Test: Invalid complexity preference raises ValueError
def test_create_player_with_invalid_complexity_range_raises_error():
    with pytest.raises(ValueError, match="complexity"):
        Player(name="Test", complexity_preference=(4.0, 2.0))
```

### Implementation Steps

1. Write tests in `tests/test_models/test_player.py`
2. Run tests to confirm they fail
3. Implement `src/boardgame_night/models/player.py`
4. Run all checks
5. Commit: `git commit -m "feat: Add Player model with validation"`

---

## Feature 3: PlaySession Model

### Overview

Create a `PlaySession` dataclass recording when a game was played.

### Specification

**File:** `src/boardgame_night/models/play_session.py`

**Class:** `PlaySession`

**Attributes:**
| Attribute | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `id` | `str` | Unique identifier | Auto-generated UUID |
| `game` | `Game` | The game played | Required |
| `players` | `frozenset[Player]` | Who played | Non-empty, within game's player range |
| `date` | `datetime` | When played | Required |
| `duration_minutes` | `int \| None` | Actual duration | Optional, > 0 if provided |
| `winner` | `Player \| None` | Who won | Must be in players if provided |
| `notes` | `str` | Session notes | Optional |

**Validation Rules:**
1. `players` must have at least `game.min_players` players
2. `players` must have at most `game.max_players` players
3. If `winner` is provided, they must be in `players`
4. If `duration_minutes` is provided, it must be > 0

### Test Cases

**File:** `tests/test_models/test_play_session.py`

```python
# Test: Create valid play session
def test_create_play_session_with_valid_data(sample_game, sample_players):
    session = PlaySession(
        game=sample_game,
        players=frozenset(sample_players[:3]),
        date=datetime.now(),
    )
    assert session.game == sample_game
    assert len(session.players) == 3

# Test: Too few players raises ValueError
def test_create_play_session_with_too_few_players_raises_error():
    game = Game(name="Test", min_players=3, max_players=4, ...)
    with pytest.raises(ValueError, match="player"):
        PlaySession(game=game, players=frozenset([player1, player2]), ...)

# Test: Winner not in players raises ValueError
def test_create_play_session_with_invalid_winner_raises_error():
    with pytest.raises(ValueError, match="winner"):
        PlaySession(..., winner=non_participant)
```

### Implementation Steps

1. Add fixtures for `sample_game` and `sample_players` to `conftest.py`
2. Write tests in `tests/test_models/test_play_session.py`
3. Run tests to confirm they fail
4. Implement `src/boardgame_night/models/play_session.py`
5. Run all checks
6. Commit: `git commit -m "feat: Add PlaySession model with validation"`

---

## Feature 4: Game Collection Service

### Overview

Create a service to manage a collection of games with CRUD operations.

### Specification

**File:** `src/boardgame_night/services/collection.py`

**Class:** `GameCollection`

**Methods:**
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `add_game` | `game: Game` | `None` | Add game to collection |
| `remove_game` | `game_name: str` | `bool` | Remove game, return True if found |
| `get_game` | `name: str` | `Game \| None` | Find game by name |
| `list_games` | | `list[Game]` | All games, sorted by name |
| `find_games` | See below | `list[Game]` | Filter games by criteria |

**`find_games` Parameters:**
- `player_count: int | None` - Filter by player count support
- `max_playtime: int | None` - Filter by maximum playtime
- `max_complexity: float | None` - Filter by maximum complexity
- `categories: set[str] | None` - Filter by category (any match)

### Test Cases

**File:** `tests/test_services/test_collection.py`

```python
# Test: Add and retrieve game
def test_add_and_get_game():
    collection = GameCollection()
    game = Game(name="Catan", ...)
    collection.add_game(game)
    assert collection.get_game("Catan") == game

# Test: Remove existing game returns True
def test_remove_existing_game_returns_true():
    collection = GameCollection()
    collection.add_game(Game(name="Catan", ...))
    assert collection.remove_game("Catan") is True
    assert collection.get_game("Catan") is None

# Test: Remove non-existent game returns False
def test_remove_nonexistent_game_returns_false():
    collection = GameCollection()
    assert collection.remove_game("Unknown") is False

# Test: find_games filters by player count
def test_find_games_filters_by_player_count():
    collection = GameCollection()
    collection.add_game(Game(name="TwoPlayer", min_players=2, max_players=2, ...))
    collection.add_game(Game(name="FourPlayer", min_players=3, max_players=4, ...))
    results = collection.find_games(player_count=2)
    assert len(results) == 1
    assert results[0].name == "TwoPlayer"

# Test: find_games filters by multiple criteria
def test_find_games_filters_by_multiple_criteria():
    # Add games with varying attributes
    # Filter by player_count=4 AND max_playtime=60
    # Verify only matching games returned
```

### Implementation Steps

1. Create `src/boardgame_night/services/__init__.py`
2. Create `tests/test_services/__init__.py`
3. Write tests in `tests/test_services/test_collection.py`
4. Run tests to confirm they fail
5. Implement `src/boardgame_night/services/collection.py`
6. Run all checks
7. Commit: `git commit -m "feat: Add GameCollection service with CRUD and filtering"`

---

## Feature 5: Game Recommendation Service

### Overview

Create a service that recommends games based on constraints and preferences.

### Specification

**File:** `src/boardgame_night/services/recommendations.py`

**Class:** `RecommendationService`

**Constructor:**
- `collection: GameCollection` - The game collection to recommend from

**Methods:**
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `recommend` | See below | `list[Game]` | Recommend games for a game night |

**`recommend` Parameters:**
- `player_count: int` - Number of players (required)
- `available_time: int | None` - Available time in minutes
- `players: list[Player] | None` - Players (for preference matching)
- `max_complexity: float | None` - Maximum complexity
- `limit: int = 5` - Maximum recommendations to return

**Algorithm:**
1. Filter games that support `player_count`
2. If `available_time` provided, filter by `fits_time_slot`
3. If `max_complexity` provided, filter by complexity
4. If `players` provided, score games by preference match
5. Sort by score (or alphabetically if no preferences)
6. Return top `limit` games

### Test Cases

**File:** `tests/test_services/test_recommendations.py`

```python
# Test: Recommend filters by player count
def test_recommend_filters_by_player_count():
    ...

# Test: Recommend respects time constraint
def test_recommend_respects_time_constraint():
    ...

# Test: Recommend considers player preferences
def test_recommend_considers_player_preferences():
    ...

# Test: Recommend returns empty list when no matches
def test_recommend_returns_empty_when_no_matches():
    ...

# Test: Recommend respects limit parameter
def test_recommend_respects_limit():
    ...
```

### Implementation Steps

1. Write tests in `tests/test_services/test_recommendations.py`
2. Run tests to confirm they fail
3. Implement `src/boardgame_night/services/recommendations.py`
4. Run all checks
5. Commit: `git commit -m "feat: Add RecommendationService with filtering and preferences"`

---

## Verification Checklist

After implementing each feature, verify:

- [ ] All tests pass: `uv run pytest`
- [ ] Code is formatted: `uv run ruff format .`
- [ ] No lint errors: `uv run ruff check .`
- [ ] Types are correct: `uv run mypy src/`
- [ ] Changes are committed with descriptive message

## Tips for AI Assistants

1. **Read CLAUDE.md first** - It contains project conventions
2. **Write tests before implementation** - Follow TDD
3. **Run all checks after each change** - Don't let errors accumulate
4. **Commit frequently** - Each successful feature = one commit
5. **Don't modify tests to pass** - Fix the implementation instead
6. **Use type hints everywhere** - mypy is in strict mode
