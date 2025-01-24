import os
from dotenv import load_dotenv
import requests
import pandas as pd

# Load environment variables from .env file
load_dotenv()

def get_historical_eps_alpha_vantage(ticker, api_key):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "EARNINGS",
        "symbol": ticker,
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if "annualEarnings" in data:
        # Get the MM-DD format of the 11th record
        mm_dd = data["annualEarnings"][-1]["fiscalDateEnding"][5:]
        # Extract EPS data and filter for the same MM-DD
        eps_data = [
            {"date": entry["fiscalDateEnding"], "eps": float(entry["reportedEPS"])}
            for entry in data["annualEarnings"][:11] if entry["fiscalDateEnding"][5:] == mm_dd
        ]
        return pd.DataFrame(eps_data)
    else:
        return None

# Read the API key from the environment variables
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
ticker = "PYPL"

# Fetch EPS data
eps_data = get_historical_eps_alpha_vantage(ticker, api_key)
print(eps_data)
