"""Member management for the library system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import date


@dataclass
class Member:
    """Represents a library member."""

    member_id: str
    name: str
    email: str
    join_date: date
    borrowed_books: List[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> Dict:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "email": self.email,
            "join_date": self.join_date.isoformat(),
            "borrowed_books": self.borrowed_books,
            "active": self.active,
        }


class MemberRegistry:
    """Manages library members."""

    def __init__(self):
        self._members: Dict[str, Member] = {}

    def register_member(self, member: Member) -> bool:
        """
        Register a new member.

        Args:
            member: Member object to register

        Returns:
            True if registered, False if member_id already exists
        """
        if member.member_id in self._members:
            return False
        self._members[member.member_id] = member
        return True

    def get_member(self, member_id: str) -> Optional[Member]:
        """
        Get a member by ID.

        Args:
            member_id: The member's ID

        Returns:
            Member object if found, None otherwise
        """
        return self._members.get(member_id)

    def deactivate_member(self, member_id: str) -> bool:
        """
        Deactivate a member account.

        Args:
            member_id: The member's ID

        Returns:
            True if deactivated, False if member not found
        """
        member = self._members.get(member_id)
        if member:
            member.active = False
            return True
        return False

    def get_active_members(self) -> List[Member]:
        """Get all active members."""
        return [member for member in self._members.values() if member.active]

    def search_by_name(self, name: str) -> List[Member]:
        """
        Search members by name (partial match).

        Args:
            name: Name or partial name to search for

        Returns:
            List of matching members
        """
        name_lower = name.lower()
        return [
            member for member in self._members.values() if name_lower in member.name.lower()
        ]
