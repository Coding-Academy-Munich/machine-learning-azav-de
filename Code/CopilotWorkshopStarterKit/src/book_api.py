"""Book data models and API for the library."""

from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, date
from enum import Enum


class BookStatus(Enum):
    """Status of a book in the library."""

    AVAILABLE = "available"
    BORROWED = "borrowed"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"


@dataclass
class Book:
    """Represents a book in the library."""

    isbn: str
    title: str
    author: str
    year: int
    genre: str
    status: BookStatus = BookStatus.AVAILABLE
    borrowed_by: Optional[str] = None
    due_date: Optional[date] = None

    def to_dict(self) -> Dict:
        return {
            "isbn": self.isbn,
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "genre": self.genre,
            "status": self.status.value,
            "borrowed_by": self.borrowed_by,
            "due_date": self.due_date.isoformat() if self.due_date else None,
        }


class BookCatalog:
    """Manages the book inventory."""

    def __init__(self):
        self._books: Dict[str, Book] = {}
        self._cache: Dict[str, tuple] = {}

    def add_book(self, book: Book) -> bool:
        """
        Add a book to the catalog.

        Args:
            book: Book object to add

        Returns:
            True if book was added, False if ISBN already exists
        """
        if book.isbn in self._books:
            return False
        self._books[book.isbn] = book
        return True

    def get_book(self, isbn: str) -> Optional[Book]:
        """
        Get a book by ISBN.

        Args:
            isbn: The ISBN of the book

        Returns:
            Book object if found, None otherwise
        """
        return self._books.get(isbn)

    def search_by_title(self, title: str) -> List[Book]:
        """
        Search books by title (partial match).

        Args:
            title: Title or partial title to search for

        Returns:
            List of matching books
        """
        title_lower = title.lower()
        return [
            book for book in self._books.values() if title_lower in book.title.lower()
        ]

    def search_by_author(self, author: str) -> List[Book]:
        """
        Search books by author (partial match).

        Args:
            author: Author name or partial name

        Returns:
            List of matching books
        """
        author_lower = author.lower()
        return [
            book
            for book in self._books.values()
            if author_lower in book.author.lower()
        ]

    def get_available_books(self) -> List[Book]:
        """Get all available books."""
        return [
            book
            for book in self._books.values()
            if book.status == BookStatus.AVAILABLE
        ]

    def get_borrowed_books(self) -> List[Book]:
        """Get all borrowed books."""
        return [
            book for book in self._books.values() if book.status == BookStatus.BORROWED
        ]

    def update_book_status(self, isbn: str, status: BookStatus) -> bool:
        """
        Update the status of a book.

        Args:
            isbn: ISBN of the book
            status: New status

        Returns:
            True if updated, False if book not found
        """
        book = self._books.get(isbn)
        if book:
            book.status = status
            return True
        return False


def create_sample_catalog() -> BookCatalog:
    """Create a catalog with sample books for testing."""
    catalog = BookCatalog()

    sample_books = [
        Book(
            "978-0-13-110362-7",
            "The C Programming Language",
            "Brian Kernighan",
            1988,
            "Programming",
        ),
        Book(
            "978-0-201-61622-4",
            "The Pragmatic Programmer",
            "Andrew Hunt",
            1999,
            "Programming",
        ),
        Book(
            "978-0-132-35088-4", "Clean Code", "Robert C. Martin", 2008, "Programming"
        ),
        Book(
            "978-0-596-52068-7",
            "JavaScript: The Good Parts",
            "Douglas Crockford",
            2008,
            "Programming",
        ),
        Book(
            "978-0-134-68599-1",
            "Effective Python",
            "Brett Slatkin",
            2019,
            "Programming",
        ),
    ]

    for book in sample_books:
        catalog.add_book(book)

    return catalog
