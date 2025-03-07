"""Module to store data models."""

from dataclasses import dataclass


@dataclass
class Config:
    """Config data."""

    modelling: dict[str, str]
    past_years: int
    future_years: int
    freq: str
