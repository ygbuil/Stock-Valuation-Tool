import os
from dotenv import load_dotenv
import requests
import pandas as pd

load_dotenv()


def f(ticker, api_key):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "INCOME_STATEMENT",
        "symbol": ticker,
        "apikey": "aaa"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    return pd.DataFrame(data["quarterlyReports"])


def get_fundamental_data(ticker, api_key):
    from alpha_vantage.fundamentaldata import FundamentalData

    # api_key = "YOUR_API_KEYs"

    # Create an instance of the FundamentalData class
    fd = FundamentalData(key=api_key, output_format='json')

    income_statement = fd.get_income_statement_quarterly(ticker)[0]
    cash_flow_statement = fd.get_cash_flow_quarterly(ticker)[0].assign(fiscalDateEdning=lambda df: pd.Timestamp(df["fiscalDateEdning"]).strftime('%Y-%m'), freeCashflow=lambda df: pd.to_numeric(df["operatingCashflow"]) - pd.to_numeric(df["capitalExpenditures"]))["fiscalDateEnding", "freeCashflow"]
    

    cash_flow_statement["freeCashflow"] = cash_flow_statement["operatingCashflow"] - cash_flow_statement["capitalExpenditures"]

    return income_statement, cash_flow_statement


# def f(ticker, api_key):
#     import requests

#     # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
#     url = f'https://www.alphavantage.co/query?function=SPLITS&symbol={ticker}&apikey={api_key}'
#     r = requests.get(url)
#     data = r.json()


# Read the API key from the environment variables
api_key = os.getenv("API_KEY_ALPHA_VANTAGE")
ticker = "PYPL"

# Fetch EPS data
eps_data = get_fundamental_data(ticker, api_key)
print(eps_data)
