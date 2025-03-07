"""Module to store data models."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    """Config data."""

    modelling: dict[str, dict[str, Any]]
    past_years: int
    future_years: int
    freq: str
