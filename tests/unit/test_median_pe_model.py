"""Test Median PE Model."""

import numpy as np
import pytest
from pytest import FixtureRequest  # noqa: PT013

from stock_valuation_tool.modelling._models import MedianPeModel


@pytest.fixture
def train_data_1() -> list[float]:
    """Train data.

    Returns:
        Train data.
    """
    return [5, 8, 9, 6, 10, 12, 15, 11, 19, 20, 20, 21]


@pytest.fixture
def median_1(train_data_1: list[float]) -> float:
    """Median value of the training data.

    Returns:
        Median value.
    """
    return float(np.median(train_data_1))


@pytest.mark.parametrize(
    ("train_data", "median"),
    [
        ("train_data_1", "median_1"),
    ],
)
def test_median_pe_model(
    train_data: str,
    median: str,
    request: FixtureRequest,
) -> None:
    """Test MedianPeModel.

    Args:
        train_data: Train data.
        median: Expected median value.
        request: FixtureRequest.
    """
    median_pe_model = MedianPeModel()
    median_pe_model.train(y_train=request.getfixturevalue(train_data))

    assert median_pe_model.median == request.getfixturevalue(median)
    assert median_pe_model.predict() == request.getfixturevalue(median)
