import yfinance as yf
import pandas as pd
from loguru import logger
from stock_valuation.exceptions import YahooFinanceError
from sklearn.linear_model import LinearRegression


def preprocess(ticker) -> pd.DataFrame:
    eps = _get_income_statement_data(ticker)
    fcf = _get_cash_flow_statement_data(ticker)
    shares_outstanding = _get_balance_sheet_data(ticker)
    prices = _load_prices(ticker, pd.Timestamp("2019-01-01"))

    data = (
        eps
        .merge(prices, on="date", how="left")
        .merge(fcf, on="date", how="left")
        .merge(shares_outstanding, on="date", how="left")
    ).assign(pe=lambda df: df["close_adj_origin_currency"] / df["eps"],)

    calculate_future_metrics(data, prices)

    return data

    


def _get_income_statement_data(ticker) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    income_stmt = stock.income_stmt
    diluted_eps = income_stmt.loc["Diluted EPS"]
    net_income = income_stmt.loc["Net Income"]

    return pd.DataFrame({
        "date": diluted_eps.index,
        "eps": diluted_eps.values,
        "net_income": net_income.values
    })


def _get_cash_flow_statement_data(ticker) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    cash_flow = stock.cash_flow
    fcf = cash_flow.loc["Free Cash Flow"]

    return pd.DataFrame({
        "date": fcf.index,
        "fcf": fcf.values
    })

def _get_balance_sheet_data(ticker) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    shares_outstanding = balance_sheet.loc["Ordinary Shares Number"]

    return pd.DataFrame({
        "date": shares_outstanding.index,
        "shares_outstanding": shares_outstanding.values
    })


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
            .assign(date=lambda df: pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d"))
        ))
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


def calculate_future_metrics(data, prices):

    X_train, y_train = list(reversed([[x.year] for x in data["date"]])), list(reversed(data["eps"]))
    model = lin_reg(X_train, y_train)

    current_price, current_year = prices["close_adj_origin_currency"].iloc[0], prices["date"].iloc[0].year

    eps_5_yrs = model.predict([[current_year + 5]])[0]

    price_5_yrs_constant_pe = model_constant_pe(data, eps_5_yrs)

    price_5_yrs_multiple_expansion = model_multiple_expansion(data, eps_5_yrs, current_year)

    

    returns = [
        {"modelling_method": "constant_pe", "price_5_yrs": price_5_yrs_constant_pe, "return_5_yrs": (price_5_yrs_constant_pe / current_price - 1)*100},
        {"modelling_method": "multiple_expansion", "price_5_yrs": price_5_yrs_multiple_expansion, "return_5_yrs": (price_5_yrs_multiple_expansion / current_price - 1)*100}
    ]
    returns = pd.DataFrame({
        "modelling_method": ["constant_pe", "multiple_expansion"],
        "price_5_yrs": [price_5_yrs_constant_pe, price_5_yrs_multiple_expansion],
        "return_5_yrs": [(price_5_yrs_constant_pe / current_price - 1)*100, (price_5_yrs_multiple_expansion / current_price - 1)*100]
    })
    

    return returns


def model_constant_pe(data, eps_5_yrs) -> float:
    price_5_yrs = eps_5_yrs * data["pe"].mean()

    return price_5_yrs

def model_multiple_expansion(data, eps_5_yrs, current_year) -> float:
    X_train, y_train = list(reversed([[x.year] for x in data["date"]])), list(reversed(data["pe"]))
    model = lin_reg(X_train, y_train)

    pe_5_yrs = model.predict([[current_year + 5]])[0]

    price_5_yrs = eps_5_yrs * pe_5_yrs

    return price_5_yrs


def lin_reg(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


print(preprocess("AAPL"))