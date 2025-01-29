import os
import requests

# Replace with your API key
api_key = os.getenv("API_KEY_EOD")

# Replace with your stock symbol and exchange
SYMBOL = "AAPL.US"  # Example: AAPL for Apple on US exchange

# EOD Financials API URL
url = f"https://eodhistoricaldata.com/api/fundamentals/{SYMBOL}?api_token={api_key}"

# Send the request
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    # Navigate the JSON response to find TTM revenue
    try:
        ttm_revenue = data["Financials"]["Income_Statement"]["ttm"]["totalRevenue"]
        print(f"TTM Revenue for {SYMBOL}: {ttm_revenue}")
    except KeyError:
        print("TTM Revenue data not found!")
else:
    print(f"Error: {response.status_code}, {response.text}")
