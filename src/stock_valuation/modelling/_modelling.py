from typing import Any
import yfinance as yf
import pandas as pd
from loguru import logger
from stock_valuation.exceptions import YahooFinanceError
from sklearn.linear_model import LinearRegression


def modelling(data: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    current_price, current_year = (
        prices["close_adj_origin_currency"].iloc[0],
        prices["date"].iloc[0].year,
    )
    year_5 = current_year + 5

    eps_5_yrs = _calculate_5_yrs_eps(data, year_5)

    pe_5_yrs_ct, pe_5_yrs_mult_exp = _calculate_5_yrs_pe(data, year_5)

    price_5_yrs_ct_pe, price_5_yrs_mult_exp = (
        eps_5_yrs * pe_5_yrs_ct,
        eps_5_yrs * pe_5_yrs_mult_exp,
    )

    returns = pd.DataFrame(
        [
            {"modelling_method": "constant_pe", "price_5_yrs": price_5_yrs_ct_pe},
            {"modelling_method": "multiple_expansion", "price_5_yrs": price_5_yrs_mult_exp},
        ]
    ).assign(
        date=year_5,
        eps=eps_5_yrs,
        return_5_yrs=lambda df: (df["price_5_yrs"] / current_price - 1) * 100,
    )["date", "modelling_method", "eps", "price_5_yrs", "return_5_yrs"]

    return returns


def _calculate_5_yrs_eps(data: pd.DataFrame, year_5: int) -> float:
    X_train, y_train = list(reversed([[x.year] for x in data["date"]])), list(reversed(data["eps"]))
    lin_reg_eps = _lin_reg(X_train, y_train)

    return lin_reg_eps.predict([[year_5]])[0]


def _lin_reg(X_train: list[list[int]], y_train: list[float]) -> LinearRegression:
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    return lin_reg


def _calculate_5_yrs_pe(data: pd.DataFrame, year_5: int) -> tuple[float, float]:
    X_train, y_train = list(reversed([[x.year] for x in data["date"]])), list(reversed(data["pe"]))
    lin_reg_pe = _lin_reg(X_train, y_train)

    pe_5_yrs = lin_reg_pe.predict([[year_5]])[0]

    return data["pe"].mean(), pe_5_yrs
