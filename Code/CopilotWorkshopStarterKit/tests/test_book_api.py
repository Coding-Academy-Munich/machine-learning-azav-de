"""Tests for the book catalog functionality."""

import pytest
from datetime import date
from src.book_api import Book, BookCatalog, BookStatus, create_sample_catalog


def test_add_book():
    """Test adding a book to the catalog."""
    catalog = BookCatalog()
    book = Book(
        isbn="978-0-123456-78-9",
        title="Test Book",
        author="Test Author",
        year=2024,
        genre="Testing",
    )

    result = catalog.add_book(book)
    assert result is True
    assert catalog.get_book("978-0-123456-78-9") == book


def test_add_duplicate_book():
    """Test that duplicate ISBNs are rejected."""
    catalog = BookCatalog()
    book1 = Book(
        isbn="978-0-123456-78-9",
        title="Test Book 1",
        author="Author 1",
        year=2024,
        genre="Testing",
    )
    book2 = Book(
        isbn="978-0-123456-78-9",
        title="Test Book 2",
        author="Author 2",
        year=2024,
        genre="Testing",
    )

    catalog.add_book(book1)
    result = catalog.add_book(book2)

    assert result is False
    assert catalog.get_book("978-0-123456-78-9") == book1


def test_search_by_title():
    """Test searching books by title."""
    catalog = create_sample_catalog()
    results = catalog.search_by_title("Python")

    assert len(results) == 1
    assert results[0].title == "Effective Python"


def test_get_available_books():
    """Test retrieving available books."""
    catalog = create_sample_catalog()
    available = catalog.get_available_books()

    assert len(available) == 5
    for book in available:
        assert book.status == BookStatus.AVAILABLE
