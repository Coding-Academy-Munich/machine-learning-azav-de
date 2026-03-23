"""Configuration settings for the library application."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the library system."""

    max_books_per_member: int = 3
    loan_period_days: int = 14
    late_fee_per_day: float = 0.50
    cache_ttl: int = 300  # seconds


DEFAULT_CONFIG = Config()
