"""Test the model selection function."""

import pytest
from pytest import FixtureRequest  # noqa: PT013

from stock_valuation_tool.modelling._modelling import _model_selection
from stock_valuation_tool.modelling._models import (
    CustomPeModel,
    ExponentialModel,
    LinReg,
    MedianPeModel,
)
from stock_valuation_tool.utils import Config

MODELLING_TYPE = "test"
CQGR = 1.0241


@pytest.fixture
def config_median() -> Config:
    """Config for median modelling.

    Returns:
        Config
    """
    return Config(
        modelling={MODELLING_TYPE: {"model": "median"}}, past_years=5, future_years=2, freq="ttm"
    )


@pytest.fixture
def config_custom_pe() -> Config:
    """Config for custom pe modelling.

    Returns:
        Config
    """
    return Config(
        modelling={MODELLING_TYPE: {"model": "custom_pe", "value": 20.0}},
        past_years=5,
        future_years=2,
        freq="ttm",
    )


@pytest.fixture
def config_linear() -> Config:
    """Config for linear regression modelling.

    Returns:
        Config
    """
    return Config(
        modelling={MODELLING_TYPE: {"model": "linear"}}, past_years=5, future_years=2, freq="ttm"
    )


@pytest.fixture
def config_exp() -> Config:
    """Config for exponential modelling.

    Returns:
        Config
    """
    return Config(
        modelling={MODELLING_TYPE: {"model": "exp"}}, past_years=5, future_years=2, freq="ttm"
    )


@pytest.fixture
def config_custom_cagr() -> Config:
    """Config for custom cagr modelling.

    Returns:
        Config
    """
    return Config(
        modelling={MODELLING_TYPE: {"model": "custom_cagr", "value": 10.0}},
        past_years=5,
        future_years=2,
        freq="ttm",
    )


@pytest.fixture
def X_train() -> list[list[int]]:  # noqa: N802
    """x_train.

    Returns:
        x_train
    """
    return [[i] for i in range(1, 6)]


@pytest.fixture
def y_train() -> list[float]:
    """y_train.

    Returns:
        y_train
    """
    return [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("config_fixture", "expected_model"),
    [
        ("config_median", MedianPeModel),
        ("config_custom_pe", CustomPeModel),
        ("config_linear", LinReg),
        ("config_exp", ExponentialModel),
        ("config_custom_cagr", ExponentialModel),
    ],
)
def test_model_selection(
    config_fixture: str,
    expected_model: type,
    X_train: list[list[int]],  # noqa: N803
    y_train: list[float],
    request: FixtureRequest,
) -> None:
    """Test _model_selection function.

    Args:
        config_fixture: The configuration fixture name.
        expected_model: Expected model type.
        X_train: Training feature data.
        y_train: Training target data.
        request: FixtureRequest.
    """
    config = request.getfixturevalue(config_fixture)
    model = _model_selection(
        config, X_train, y_train, past_periods=5, modelling_type=MODELLING_TYPE
    )

    assert isinstance(model, expected_model)

    if config.modelling[MODELLING_TYPE]["model"] == "custom_cagr":
        assert model.cqgr == CQGR  # type: ignore
