```instructions
# Copilot Instructions for BookLibrary

## Project Overview
BookLibrary is a Python library management system for tracking books, members, and borrowing transactions.

## Code Style Guidelines
- Use type hints for all function parameters and return values
- Follow PEP 8 naming conventions
- Use dataclasses for data structures where appropriate
- Write docstrings for all public functions and classes
- Prefer explicit over implicit

## Architecture
- `src/config.py` - Configuration management
- `src/book_api.py` - Book models and catalog management
- `src/member_manager.py` - Member registration and management
- `src/library.py` - Main library operations (borrow, return)

## Testing
- Use pytest for all tests
- Place tests in the `tests/` directory
- Test both success and failure cases
- Mock external dependencies if needed

## Error Handling
- Validate all input parameters
- Return False or None for failed operations
- Use descriptive variable names
- Consider edge cases (empty lists, invalid IDs)

## When generating code
- Always add type hints
- Include docstrings with Args, Returns sections
- Handle None values appropriately
- Use enums for status fields

```
