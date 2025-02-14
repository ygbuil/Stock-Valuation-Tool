"""Preprocessing module."""

import json
from pathlib import Path

import pandas as pd
import yfinance as yf  # type: ignore
from alpha_vantage.fundamentaldata import FundamentalData  # type: ignore
from loguru import logger

from stock_valuation_tool.exceptions import YahooFinanceError
from stock_valuation_tool.utils import Config


def preprocess(
    ticker: str, benchmark: str, data_source: str
) -> tuple[Config, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = _load_config(Path("config/config.json"))
    start_date = pd.Timestamp.today().normalize() - pd.DateOffset(years=config.past_years)
    prices = _load_prices(ticker, start_date)
    benchmark_prices = _load_prices(benchmark, start_date)

    if data_source == "api":
        income_statement = _get_fundamental_data(ticker)
        income_statement.to_csv("income_statement.csv", index=False)
    else:
        income_statement = pd.read_csv("income_statement.csv").assign(
            date=lambda df: pd.to_datetime(df["date"])
        )

    if config.freq == "ttm":
        income_statement = _calculate_ttm(data=income_statement)

    past_fundamentals = (
        income_statement[income_statement["date"] >= start_date].merge(
            prices, on="date", how="left"
        )
    ).assign(
        pe=lambda df: df["close_adj_origin_currency"] / df["eps"],
    )

    return config, past_fundamentals, prices, benchmark_prices


def _load_config(config_path: Path) -> Config:
    """Load config.json.

    Args:
        config_path: Config file name.

    Returns:
        Config dataclass with the info of config.json.
    """
    with (config_path).open() as file:
        return Config(**json.load(file))


def _get_fundamental_data(ticker: str) -> pd.DataFrame:
    fd = FundamentalData(key="None", output_format="json")

    income_statement: pd.DataFrame = (
        fd.get_income_statement_quarterly(ticker)[0][["fiscalDateEnding", "netIncome"]]
        .rename({"fiscalDateEnding": "date", "netIncome": "net_income"}, axis=1)
        .assign(
            date=lambda df: pd.to_datetime(df["date"]),
            net_income=lambda df: pd.to_numeric(df["net_income"]),
        )
    )
    balance_sheet = (
        fd.get_balance_sheet_quarterly(ticker)[0][
            ["fiscalDateEnding", "commonStockSharesOutstanding"]
        ]
        .rename(
            {"fiscalDateEnding": "date", "commonStockSharesOutstanding": "shares_outstanding"},
            axis=1,
        )
        .assign(
            date=lambda df: pd.to_datetime(df["date"]),
            shares_outstanding=lambda df: pd.to_numeric(df["shares_outstanding"]),
        )
    )

    return (
        income_statement.merge(balance_sheet, on="date", how="left")
        .dropna()
        .assign(eps=lambda df: df["net_income"] / df["shares_outstanding"])[["date", "eps"]]
    )


def _get_cash_flow_data(ticker: str) -> pd.DataFrame:
    fd = FundamentalData(key="None", output_format="json")

    cash_flow_statement: pd.DataFrame = fd.get_cash_flow_quarterly(ticker)[0].assign(
        fiscalDateEdning=lambda df: pd.Timestamp(df["fiscalDateEdning"]).strftime("%Y-%m"),
        freeCashflow=lambda df: pd.to_numeric(df["operatingCashflow"])
        - pd.to_numeric(df["capitalExpenditures"]),
    )["fiscalDateEnding", "freeCashflow"]
    cash_flow_statement["freeCashflow"] = (
        cash_flow_statement["operatingCashflow"] - cash_flow_statement["capitalExpenditures"]
    )

    return cash_flow_statement


def _calculate_ttm(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the trailing twelve months (TTM) for a given dataframe.

    Args:
        data: Dataframe with the historical data.

    Returns:
        Dataframe with the TTM data.
    """
    data["eps"] = data["eps"][::-1].rolling(4).sum()[::-1]

    return data.dropna()


def _load_prices(
    ticker: str,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load the following daily data at market close for a given ticker:
        - Unadjusted asset price.
        - Stock splits.
        - Dividends (at Ex-Dividend Date).

    Args:
        ticker: Asset ticker
        start_date: Start date to load the data.

    Raises:
        YahooFinanceError: Something went wrong with the Yahoo Finance API.

    Returns:
        Dataframe with the historical asset price and stock splits.
    """
    logger.info(f"Loading historical data for {ticker}")
    full_date_range = pd.DataFrame(
        {
            "date": reversed(
                pd.date_range(
                    start=start_date,
                    end=pd.Timestamp.today().normalize(),
                    freq="D",
                ),
            ),
        },
    )

    try:
        asset = yf.Ticker(ticker)
        asset_data = (
            asset.history(start=start_date)[["Close", "Volume"]]
            .sort_index(ascending=False)
            .reset_index()
            .rename(
                columns={
                    "Date": "date",
                    "Close": "close_adj_origin_currency",
                },
            )
            .assign(date=lambda df: pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d")))
        )
    except Exception as exc:
        msg = f"Something went wrong retrieving Yahoo Finance data for ticker {ticker}: {exc}"
        raise YahooFinanceError(msg) from exc

    return full_date_range.merge(
        asset_data,
        "left",
        on="date",
    ).assign(
        close_adj_origin_currency=lambda df: df["close_adj_origin_currency"].bfill().ffill(),
    )[
        [
            "date",
            "close_adj_origin_currency",
        ]
    ]
