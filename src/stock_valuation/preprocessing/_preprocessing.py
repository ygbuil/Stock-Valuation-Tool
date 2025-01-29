import yfinance as yf
import pandas as pd
from loguru import logger
from stock_valuation.exceptions import YahooFinanceError
from sklearn.linear_model import LinearRegression


def preprocess(ticker) -> pd.DataFrame:
    income_statement = _get_income_statement_data(ticker)
    cash_flow_statement = _get_cash_flow_statement_data(ticker)
    balance_sheet = _get_balance_sheet_data(ticker)
    prices = _load_prices(ticker, pd.Timestamp("2019-01-01"))

    data = (
        income_statement.merge(prices, on="date", how="left")
        # .merge(cash_flow_statement, on="date", how="left")
        # .merge(balance_sheet, on="date", how="left")
    ).assign(
        pe=lambda df: df["close_adj_origin_currency"] / df["eps"],
    )

    return data, prices


def _get_income_statement_data(ticker) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    income_stmt = stock.income_stmt
    diluted_eps = income_stmt.loc["Diluted EPS"]
    # net_income = income_stmt.loc["Net Income"]

    return pd.DataFrame(
        {"date": diluted_eps.index, "eps": diluted_eps.values}
        #  "net_income": net_income.values}
    )


def _get_cash_flow_statement_data(ticker) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    cash_flow_statement = stock.cash_flow
    cash_flow_statement = cash_flow_statement.loc["Free Cash Flow"]

    return pd.DataFrame(
        {"date": cash_flow_statement.index, "free_cash_flow": cash_flow_statement.values}
    )


def _get_balance_sheet_data(ticker) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    shares_outstanding = balance_sheet.loc["Ordinary Shares Number"]

    return pd.DataFrame(
        {"date": shares_outstanding.index, "shares_outstanding": shares_outstanding.values}
    )


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
