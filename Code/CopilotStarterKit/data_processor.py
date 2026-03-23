"""
Data Processor Module - CopilotStarterKit

This module contains various algorithms and data processing functions
that you can use to practice with GitHub Copilot:

1. Select any function and use /explain to understand it
2. Select code with potential issues and use /fix
3. Generate tests with /tests
4. Add documentation with /doc

Try these exercises with Copilot Chat!
"""

from typing import Any
from collections import defaultdict
import re


def quicksort(arr: list[int]) -> list[int]:
    """Sort a list using the quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def binary_search(arr: list[int], target: int) -> int:
    """Find target in sorted array, return index or -1."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def merge_sorted_lists(list1: list[int], list2: list[int]) -> list[int]:
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result


def find_duplicates(items: list[Any]) -> list[Any]:
    """Find all duplicate items in a list."""
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


def word_frequency(text: str) -> dict[str, int]:
    """Count frequency of each word in text."""
    words = re.findall(r'\b\w+\b', text.lower())
    frequency = defaultdict(int)
    for word in words:
        frequency[word] += 1
    return dict(frequency)


def flatten_nested_list(nested: list) -> list:
    """Flatten a nested list structure."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested_list(item))
        else:
            result.append(item)
    return result


def is_valid_email(email: str) -> bool:
    """Check if email address has valid format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def group_by_key(items: list[dict], key: str) -> dict[Any, list[dict]]:
    """Group list of dictionaries by a specific key."""
    grouped = defaultdict(list)
    for item in items:
        if key in item:
            grouped[item[key]].append(item)
    return dict(grouped)


def calculate_statistics(numbers: list[float]) -> dict[str, float]:
    """Calculate basic statistics for a list of numbers."""
    if not numbers:
        return {"count": 0, "sum": 0, "mean": 0, "min": 0, "max": 0}

    n = len(numbers)
    total = sum(numbers)
    mean = total / n
    sorted_nums = sorted(numbers)

    if n % 2 == 0:
        median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    else:
        median = sorted_nums[n // 2]

    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = variance ** 0.5

    return {
        "count": n,
        "sum": total,
        "mean": mean,
        "median": median,
        "min": min(numbers),
        "max": max(numbers),
        "std_dev": std_dev,
    }


class LRUCache:
    """Least Recently Used (LRU) Cache implementation."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[Any, Any] = {}
        self.order: list[Any] = []

    def get(self, key: Any) -> Any | None:
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key: Any, value: Any) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]

        self.cache[key] = value
        self.order.append(key)


if __name__ == "__main__":
    # Example usage - try explaining these with Copilot!
    print("QuickSort:", quicksort([3, 6, 8, 10, 1, 2, 1]))
    print("Binary Search:", binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9], 5))
    print("Duplicates:", find_duplicates([1, 2, 3, 2, 4, 3, 5]))
    print("Word Freq:", word_frequency("hello world hello python world"))
