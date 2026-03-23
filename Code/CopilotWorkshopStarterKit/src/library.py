"""Main library management system."""

from typing import Optional, List
from datetime import date, timedelta
from .book_api import BookCatalog, Book, BookStatus, create_sample_catalog
from .member_manager import MemberRegistry, Member
from .config import Config


class Library:
    """Main library management class."""

    def __init__(self, config: Config):
        self.config = config
        self.catalog = BookCatalog()
        self.members = MemberRegistry()

    def borrow_book(self, isbn: str, member_id: str) -> bool:
        """
        Process a book borrowing transaction.

        Args:
            isbn: ISBN of the book to borrow
            member_id: ID of the member borrowing the book

        Returns:
            True if successful, False otherwise
        """
        book = self.catalog.get_book(isbn)
        member = self.members.get_member(member_id)

        if not book or not member:
            return False

        if book.status != BookStatus.AVAILABLE:
            return False

        if not member.active:
            return False

        if len(member.borrowed_books) >= self.config.max_books_per_member:
            return False

        # Process the borrowing
        book.status = BookStatus.BORROWED
        book.borrowed_by = member_id
        book.due_date = date.today() + timedelta(days=self.config.loan_period_days)
        member.borrowed_books.append(isbn)

        return True

    def return_book(self, isbn: str, member_id: str) -> Optional[float]:
        """
        Process a book return transaction.

        Args:
            isbn: ISBN of the book to return
            member_id: ID of the member returning the book

        Returns:
            Late fee amount if applicable, 0.0 if on time, None if error
        """
        book = self.catalog.get_book(isbn)
        member = self.members.get_member(member_id)

        if not book or not member:
            return None

        if book.status != BookStatus.BORROWED or book.borrowed_by != member_id:
            return None

        if isbn not in member.borrowed_books:
            return None

        # Calculate late fee
        late_fee = 0.0
        if book.due_date and date.today() > book.due_date:
            days_late = (date.today() - book.due_date).days
            late_fee = days_late * self.config.late_fee_per_day

        # Process the return
        book.status = BookStatus.AVAILABLE
        book.borrowed_by = None
        book.due_date = None
        member.borrowed_books.remove(isbn)

        return late_fee

    def get_overdue_books(self) -> List[Book]:
        """Get all books that are overdue."""
        today = date.today()
        overdue = []

        for book in self.catalog.get_borrowed_books():
            if book.due_date and today > book.due_date:
                overdue.append(book)

        return overdue

    def get_member_borrowed_books(self, member_id: str) -> List[Book]:
        """
        Get all books borrowed by a specific member.

        Args:
            member_id: ID of the member

        Returns:
            List of books borrowed by the member
        """
        member = self.members.get_member(member_id)
        if not member:
            return []

        books = []
        for isbn in member.borrowed_books:
            book = self.catalog.get_book(isbn)
            if book:
                books.append(book)

        return books

    def get_library_statistics(self) -> dict:
        """Get statistics about the library."""
        all_books = len(self.catalog._books)
        available = len(self.catalog.get_available_books())
        borrowed = len(self.catalog.get_borrowed_books())
        members = len(self.members._members)
        active_members = len(self.members.get_active_members())
        overdue = len(self.get_overdue_books())

        return {
            "total_books": all_books,
            "available_books": available,
            "borrowed_books": borrowed,
            "total_members": members,
            "active_members": active_members,
            "overdue_books": overdue,
        }


def main():
    """Main entry point for the library system."""
    from .config import DEFAULT_CONFIG

    library = Library(DEFAULT_CONFIG)

    # Load sample catalog
    library.catalog = create_sample_catalog()

    # Add sample members
    member1 = Member(
        member_id="M001",
        name="Alice Smith",
        email="alice@example.com",
        join_date=date(2024, 1, 15),
    )
    member2 = Member(
        member_id="M002",
        name="Bob Johnson",
        email="bob@example.com",
        join_date=date(2024, 2, 20),
    )

    library.members.register_member(member1)
    library.members.register_member(member2)

    # Demo usage
    print("=== Library Management System ===\n")

    print("Available books:")
    for book in library.catalog.get_available_books():
        print(f"  - {book.title} by {book.author}")
    print()

    # Simulate borrowing
    success = library.borrow_book("978-0-13-110362-7", "M001")
    if success:
        print("Alice borrowed 'The C Programming Language'\n")

    # Show statistics
    stats = library.get_library_statistics()
    print("Library Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
