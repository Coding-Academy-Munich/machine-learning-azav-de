# BookLibrary - Copilot Workshop Starter Kit

A simple book library management application for practicing GitHub Copilot's advanced features.

## Features

- Book inventory management
- Member registration and tracking
- Book borrowing and returns
- Library statistics and analytics

## Project Structure

```text
CopilotWorkshopStarterKit/
  src/
    config.py         # Configuration settings
    book_api.py       # Book data models and API
    member_manager.py # Member management
    library.py        # Main library logic
  tests/
    test_book_api.py
  .github/
    copilot-instructions.md  # Custom Copilot instructions
```

## Usage with Copilot

This project is designed for practicing:

1. **@workspace** - Understanding the codebase structure
2. **Copilot Edits** - Adding features across multiple files
3. **Custom Instructions** - Following project-specific guidelines
4. **Agent Mode** - Implementing complex features autonomously

## Running

```bash
cd CopilotWorkshopStarterKit
python -m src.library
```

## Testing

```bash
pytest tests/
```

## Workshop Tasks

Students will practice:

- Adding validation and error handling
- Implementing new features (e.g., book reservations)
- Creating custom instructions
- Using chat participants effectively

