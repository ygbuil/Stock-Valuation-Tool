import requests
import os

# Set your API key from environment variables
api_key = os.getenv("API_KEY_FMP")

# Define the stock symbol
symbol = "AAPL"  # Example: Apple Inc.

# Construct the API URL for quarterly income statements
url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=quarter&apikey={api_key}"

# Make the API request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    
    # Display the last four quarters
    for report in data[:4]:  # Get the last 4 quarters
        print(f"Date: {report['date']}")
        print(f"Total Revenue: {report['revenue']}")
        print(f"Gross Profit: {report['grossProfit']}")
        print(f"Operating Income: {report['operatingIncome']}")
        print(f"Net Income: {report['netIncome']}")
        print("-" * 40)
else:
    print(f"Error: {response.status_code}, {response.text}")
