"""Preprocessing module."""

import pandas as pd
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from loguru import logger

from stock_valuation.exceptions import YahooFinanceError


def preprocess(ticker: str, benchmark: str, past_years: str, freq: str) -> pd.DataFrame:
    start_date = pd.Timestamp.today().normalize() - pd.DateOffset(years=past_years)
    prices = _load_prices(ticker, start_date)
    benchmark_prices = _load_prices(benchmark, start_date)

    # income_statement = _get_fundamental_data(ticker)
    # income_statement.to_csv("income_statement.csv", index=False)
    income_statement = pd.read_csv("income_statement.csv").assign(
        date=lambda df: pd.to_datetime(df["date"])
    )
    if freq == "ttm":
        income_statement = _calculate_ttm(data=income_statement)
    income_statement = income_statement[income_statement["date"] >= start_date]

    data = (income_statement.merge(prices, on="date", how="left")).assign(
        pe=lambda df: df["close_adj_origin_currency"] / df["eps"],
    )

    return data, prices, benchmark_prices


def _get_income_statement_data(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    income_stmt = stock.income_stmt
    diluted_eps = income_stmt.loc["Diluted EPS"]

    return pd.DataFrame({"date": diluted_eps.index, "eps": diluted_eps.to_numpy()})


def _get_cash_flow_statement_data(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    cash_flow_statement = stock.cash_flow
    cash_flow_statement = cash_flow_statement.loc["Free Cash Flow"]

    return pd.DataFrame(
        {"date": cash_flow_statement.index, "free_cash_flow": cash_flow_statement.to_numpy()}
    )


def _get_balance_sheet_data(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    shares_outstanding = balance_sheet.loc["Ordinary Shares Number"]

    return pd.DataFrame(
        {"date": shares_outstanding.index, "shares_outstanding": shares_outstanding.to_numpy()}
    )


def _get_fundamental_data(ticker: str) -> pd.DataFrame:
    fd = FundamentalData(key="None", output_format="json")

    income_statement = (
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

    # cash_flow_statement = fd.get_cash_flow_quarterly(ticker)[0].assign(
    #     fiscalDateEdning=lambda df: pd.Timestamp(df["fiscalDateEdning"]).strftime("%Y-%m"),
    #     freeCashflow=lambda df: pd.to_numeric(df["operatingCashflow"])
    #     - pd.to_numeric(df["capitalExpenditures"]),
    # )["fiscalDateEnding", "freeCashflow"]
    # cash_flow_statement["freeCashflow"] = (
    #     cash_flow_statement["operatingCashflow"] - cash_flow_statement["capitalExpenditures"]
    # )

    return (
        income_statement.merge(balance_sheet, on="date", how="left")
        .dropna()
        .assign(eps=lambda df: df["net_income"] / df["shares_outstanding"])[["date", "eps"]]
    )


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

    asset_data = full_date_range.merge(
        asset_data,
        "left",
        on="date",
    ).assign(
        close_adj_origin_currency=lambda df: df["close_adj_origin_currency"].bfill().ffill(),
    )

    return asset_data[
        [
            "date",
            "close_adj_origin_currency",
        ]
    ]
