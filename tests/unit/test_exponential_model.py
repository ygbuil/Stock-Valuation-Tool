"""Test ExponentialModel."""

import pytest
from pytest import FixtureRequest  # noqa: PT013

from stock_valuation_tool.modelling._modelling import ExponentialModel


@pytest.fixture
def train_data_1() -> list[float]:
    """Train data.

    Returns:
        Train data.
    """
    return [5, 8, 9, 6, 10, 12, 15, 11, 19, 20, 20, 21]


@pytest.fixture
def cqgr_1() -> float:
    """Compound quarterly growth rate.

    Returns:
        Compound quarterly growth rate.
    """
    return 1.1270


@pytest.fixture
def prediction_1() -> list[float]:
    """Predictions.

    Returns:
        Predictions.
    """
    return [23.667, 26.6727, 30.0601, 33.8778]


@pytest.mark.parametrize(
    ("train_data", "cqgr", "prediction"),
    [
        ("train_data_1", "cqgr_1", "prediction_1"),
    ],
)
def test_exponential_model(
    train_data: str,
    cqgr: str,
    prediction: str,
    request: FixtureRequest,
) -> None:
    """Test ExponentialModel.

    Args:
        train_data: Train data.
        cqgr: Compound quarterly growth rate.
        prediction: Predictions.
        request: FixtureRequest.
    """
    exp_model = ExponentialModel()
    exp_model.train(y_train=request.getfixturevalue(train_data))

    assert exp_model.cqgr == request.getfixturevalue(cqgr)

    assert exp_model.predict(4) == request.getfixturevalue(prediction)
