# Copilot StarterKit

Practice materials for learning GitHub Copilot features.

## Files

### data_processor.py
Contains various algorithms and data processing functions:
- Sorting algorithms (quicksort)
- Search algorithms (binary search)
- Data transformations
- Statistics calculations
- LRU Cache implementation

**Practice:**
- Select functions and use `/explain` in Copilot Chat
- Try `/tests` to generate unit tests
- Use `/doc` to add documentation

### exercises.py
Contains TODO comments and function stubs for practicing inline completion:
- Basic functions (factorial, palindrome)
- String operations
- Data processing
- File operations
- Class implementation
- Error handling
- Advanced algorithms

**Practice:**
- Position cursor after TODO comments
- Wait for Copilot suggestions (gray text)
- Accept with Tab, reject with Esc
- Try alternatives with Alt+] / Alt+[

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Accept suggestion | `Tab` |
| Reject suggestion | `Esc` |
| Next suggestion | `Alt+]` |
| Previous suggestion | `Alt+[` |
| Accept next word | `Ctrl+→` |
| Open inline chat | `Ctrl+I` |
| Open chat sidebar | `Ctrl+Alt+I` |

## Copilot Chat Commands

- `/explain` - Explain selected code
- `/fix` - Find and fix bugs
- `/tests` - Generate unit tests
- `/doc` - Add documentation
- `/simplify` - Simplify code

## Tips

1. **Better context = better suggestions**
   - Add type hints
   - Write descriptive function names
   - Add docstrings with examples

2. **Don't trust blindly**
   - Always review generated code
   - Test the output
   - Check for security issues

3. **Try multiple suggestions**
   - Press Alt+] to see alternatives
   - Different suggestions may be better for your use case
