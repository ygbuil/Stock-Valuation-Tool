"""Test Custom P/E Model."""

import pytest
from pytest import FixtureRequest  # noqa: PT013

from stock_valuation_tool.modelling._models import CustomPeModel


@pytest.fixture
def custom_pe_1() -> float:
    """Custom P/E value.

    Returns:
        Custom P/E value.
    """
    return 15.5


@pytest.mark.parametrize(
    ("custom_pe"),
    [
        ("custom_pe_1"),
    ],
)
def test_custom_pe_model(
    custom_pe: str,
    request: FixtureRequest,
) -> None:
    """Test CustomPeModel.

    Args:
        custom_pe: Custom P/E value.
        request: FixtureRequest.
    """
    custom_pe_model = CustomPeModel()
    custom_pe_model.train(custom_pe=request.getfixturevalue(custom_pe))

    assert custom_pe_model.custom_pe == request.getfixturevalue(custom_pe)
    assert custom_pe_model.predict() == request.getfixturevalue(custom_pe)
