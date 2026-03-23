"""
Exercises Module - CopilotStarterKit

This module contains TODO comments and function stubs for practicing
GitHub Copilot's inline code completion.

Instructions:
1. Position your cursor after a TODO comment or function signature
2. Wait 2-3 seconds for Copilot to suggest code
3. Press Tab to accept, or Alt+] for alternative suggestions
4. Review the generated code critically!

Tips:
- More specific comments lead to better suggestions
- Type hints help Copilot understand your intent
- Check multiple suggestions with Alt+] / Alt+[
"""


# =============================================================================
# EXERCISE 1: Basic Functions
# Try typing the function signature and let Copilot complete it
# =============================================================================

# TODO: Function to calculate the factorial of a number


# TODO: Function to check if a string is a palindrome


# TODO: Function to find the maximum value in a list


# =============================================================================
# EXERCISE 2: String Operations
# Position cursor after the docstring and let Copilot implement
# =============================================================================

def reverse_words(sentence: str) -> str:
    """Reverse the order of words in a sentence.

    Example: "hello world" -> "world hello"
    """
    # Let Copilot complete this...
    pass


def count_vowels(text: str) -> int:
    """Count the number of vowels (a, e, i, o, u) in text.

    Should be case-insensitive.
    """
    # Let Copilot complete this...
    pass


def remove_duplicates_preserve_order(items: list) -> list:
    """Remove duplicates from list while preserving original order.

    Example: [1, 2, 2, 3, 1, 4] -> [1, 2, 3, 4]
    """
    # Let Copilot complete this...
    pass


# =============================================================================
# EXERCISE 3: Data Processing
# These have detailed docstrings - see how Copilot uses the context
# =============================================================================

def calculate_average(numbers: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        numbers: List of numbers to average

    Returns:
        The arithmetic mean

    Raises:
        ValueError: If the list is empty

    Example:
        >>> calculate_average([1, 2, 3, 4, 5])
        3.0
    """
    # Let Copilot complete this...
    pass


def find_common_elements(list1: list, list2: list) -> list:
    """Find elements that appear in both lists.

    Args:
        list1: First list
        list2: Second list

    Returns:
        List of common elements (no duplicates)

    Example:
        >>> find_common_elements([1, 2, 3], [2, 3, 4])
        [2, 3]
    """
    # Let Copilot complete this...
    pass


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size.

    Args:
        items: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    # Let Copilot complete this...
    pass


# =============================================================================
# EXERCISE 4: File Operations
# Try generating these functions with more complex logic
# =============================================================================

# TODO: Function to read a CSV file and return a list of dictionaries


# TODO: Function to write a list of dictionaries to a JSON file


# TODO: Function to count lines in a text file


# =============================================================================
# EXERCISE 5: Class Implementation
# Start with the class definition and let Copilot help with methods
# =============================================================================

class Stack:
    """A simple stack implementation using a list.

    Methods:
        push(item): Add item to top of stack
        pop(): Remove and return top item
        peek(): Return top item without removing
        is_empty(): Check if stack is empty
        size(): Return number of items
    """

    def __init__(self):
        """Initialize an empty stack."""
        # Let Copilot complete...
        pass

    # TODO: Implement push method

    # TODO: Implement pop method

    # TODO: Implement peek method

    # TODO: Implement is_empty method

    # TODO: Implement size method


# =============================================================================
# EXERCISE 6: Error Handling
# These need proper error handling - see if Copilot adds it
# =============================================================================

def safe_divide(a: float, b: float) -> float:
    """Safely divide a by b.

    Returns 0 if division by zero would occur.
    """
    # Let Copilot complete this...
    pass


def parse_integer(text: str) -> int | None:
    """Parse string to integer, return None if invalid."""
    # Let Copilot complete this...
    pass


# =============================================================================
# EXERCISE 7: Advanced - Algorithm Implementation
# Try these more complex algorithms
# =============================================================================

# TODO: Function to check if two strings are anagrams


# TODO: Function to find the longest common prefix in a list of strings


# TODO: Function to generate all permutations of a string


# =============================================================================
# BONUS: Test Generation
# Select any completed function above and use /tests in Copilot Chat
# to generate unit tests!
# =============================================================================
