"""Unit tests for the LinReg model."""

import numpy as np
import pytest
from pytest import FixtureRequest  # noqa: PT013

from stock_valuation_tool.modelling._models import LinReg


@pytest.fixture
def X_train_1() -> list[list[int]]:  # noqa: N802
    """Training feature data (first test case)."""
    return [[i] for i in range(1, 6)]


@pytest.fixture
def y_train_1() -> list[float]:
    """Training target data (first test case)."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def X_train_2() -> list[list[int]]:  # noqa: N802
    """Training feature data (second test case)."""
    return [[i] for i in range(1, 6)]


@pytest.fixture
def y_train_2() -> list[float]:
    """Training target data (second test case)."""
    return [-15, -10, -5, 0, 5]


@pytest.fixture
def test_values() -> list[list[int]]:
    """Values to predict."""
    return [[6], [7]]


@pytest.fixture
def expected_predictions_1() -> list[float]:
    """Expected predictions for first test case."""
    return [6.0, 7.0]


@pytest.fixture
def expected_predictions_2() -> list[float]:
    """Expected predictions for second test case."""
    return [10.0, 15.0]


@pytest.mark.parametrize(
    ("X_train", "y_train", "expected_predictions"),
    [
        ("X_train_1", "y_train_1", "expected_predictions_1"),
        ("X_train_2", "y_train_2", "expected_predictions_2"),
    ],
)
def test_lin_reg(
    X_train: str,  # noqa: N803
    y_train: str,
    expected_predictions: str,
    test_values: list[list[int]],
    request: FixtureRequest,
) -> None:
    """Test LinReg model.

    Args:
        X_train: Training feature data.
        y_train: Training target data.
        expected_predictions: Expected predictions.
        test_values: Values to predict.
        request: FixtureRequest.
    """
    lin_reg_model = LinReg()
    lin_reg_model.train(
        X_train=request.getfixturevalue(X_train),
        y_train=request.getfixturevalue(y_train),
    )

    predictions = lin_reg_model.predict(test_values)
    np.testing.assert_almost_equal(
        predictions, request.getfixturevalue(expected_predictions), decimal=5
    )
