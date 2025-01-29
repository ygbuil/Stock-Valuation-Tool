from typing import Any
import yfinance as yf
import pandas as pd
from loguru import logger
from stock_valuation.exceptions import YahooFinanceError
from sklearn.linear_model import LinearRegression


def modelling(data: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    freq = "yearly"
    current_price, year_month_current = (
        prices["close_adj_origin_currency"].iloc[0],
        data["date"].iloc[0],
    )
    year_month_5_yrs = year_month_current + pd.DateOffset(years=5)

    data_and_pred = _calculate_5_yrs_eps_and_pe(data, freq)

    eps_5_yrs, pe_5_yrs_ct, pe_5_yrs_mult_exp = (
        data_and_pred["eps"].iloc[0],
        data_and_pred["pe_ct"].iloc[0],
        data_and_pred["pe_exp"].iloc[0],
    )
    price_5_yrs_ct_pe, price_5_yrs_mult_exp = (
        eps_5_yrs * pe_5_yrs_ct,
        eps_5_yrs * pe_5_yrs_mult_exp,
    )

    returns = pd.DataFrame(
        [
            {
                "date": data_and_pred["date"].iloc[0],
                "return_pe_ct": (
                    data_and_pred["close_adj_origin_currency_pe_ct"].iloc[0] / current_price - 1
                )
                * 100,
                "return_pe_exp": (
                    data_and_pred["close_adj_origin_currency_pe_exp"].iloc[0] / current_price - 1
                )
                * 100,
            }
        ]
    )
    return returns


def _calculate_5_yrs_eps_and_pe(data: pd.DataFrame, freq: str) -> float:
    n_data_points = len(data)
    pe_ct = data["pe"].mean()

    X_train, y_train = [[x] for x in range(n_data_points)], list(reversed(data["eps"]))
    lin_reg_eps = _lin_reg(X_train, y_train)

    X_train, y_train = [[x] for x in range(n_data_points)], list(reversed(data["pe"]))
    lin_reg_pe = _lin_reg(X_train, y_train)

    latest_date = data["date"].iloc[0]
    pred = []
    for X_pred in range(
        n_data_points, n_data_points + 5 if freq == "yearly" else n_data_points + 5 * 4
    ):
        latest_date = (
            latest_date + pd.DateOffset(years=1)
            if freq == "yearly"
            else latest_date + pd.DateOffset(months=3)
        )

        eps_pred = lin_reg_eps.predict([[X_pred]])[0]
        pe_exp_pred = lin_reg_pe.predict([[X_pred]])[0]

        pred.append(
            {
                "date": latest_date,
                "eps": eps_pred,
                "close_adj_origin_currency_pe_ct": eps_pred * pe_ct,
                "close_adj_origin_currency_pe_exp": eps_pred * pe_exp_pred,
                "pe_ct": pe_ct,
                "pe_exp": pe_exp_pred,
            }
        )

    return pd.concat(
        [
            pd.DataFrame(reversed(pred)),
            data.assign(
                close_adj_origin_currency_pe_ct=data["close_adj_origin_currency"],
                close_adj_origin_currency_pe_exp=data["close_adj_origin_currency"],
                pe_ct=data["pe"],
                pe_exp=data["pe"],
            ).drop(columns=["pe", "close_adj_origin_currency"]),
        ],
        ignore_index=True,
    )


def _lin_reg(X_train: list[list[int]], y_train: list[float]) -> LinearRegression:
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    return lin_reg
