---
description: Add unit tests for the selected code
agent: agent
model: Claude Sonnet 4.5 (copilot)
tools: ['read', 'edit', 'execute', 'search', 'search/codebase', 'web/githubRepo']
---
Add unit tests for the selected code.

- Use pytest as testing framework.
- Ensure that the tests cover various edge cases and validate the expected behavior of the code.
- Apply best practices for writing unit tests in Python.

Reference: #file:tests/conftest.py
