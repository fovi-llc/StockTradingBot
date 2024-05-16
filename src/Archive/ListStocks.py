import yfinance as yf
import pandas as pd

# URL for the NASDAQ symbol directory CSV
url = "https://www.nasdaq.com/market-activity/stocks/screener"

# Read the CSV file from the URL
symbols = pd.read_csv("nasdaq_tickers.csv")
print(symbols)
exit()

# Loop through the tickers
for ticker in symbols['Symbol']:
    stock = yf.Ticker(ticker)
    try:
        print(ticker, stock.info['shortName'])
    except KeyError:
        print(f"Data not available for ticker: {ticker}")
