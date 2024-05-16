 
import tqdm
import numpy as np
import pandas as pd
import yfinance as yf

tickers = []
#tickers.append(best_stock)
tickers.append("AAPL")

for ticker in tickers:
    # Download historical data for the ticker
    #data = yf.download(ticker, start=start_date, end=end_date)
    #data = yf.download(ticker, period="10y", interval="1d")
    #data = yf.download(ticker, period="60d", interval="90m")
    data = yf.download(ticker, period="60d", interval="60m")
    #data = yf.download(ticker, period="730d", interval="60m")
    #data = yf.download(ticker, period="60d", interval="30m")
    #data = yf.download(ticker, period="60d", interval="15m")
    #data = yf.download(ticker, period="60d", interval="5m")
    #data = yf.download(ticker, period="7d", interval="1m")

    # Calculate the daily percentage change
    close = data['Close']
    diff = data['Close'].diff(1)

    df = pd.DataFrame({
        ticker+'_close': close,
        ticker+'_diff': diff,
    })

    df.reset_index(inplace=True)
    #df['Datetime'] = df['Datetime'].dt.tz_localize(None)


print(df)


close = df[tickers[0]+'_close']
print(close)


profit = 0
share_price = []
for i in range(1, len(close)):
    curr = close[i]
    prev = close[i-1]

    diff = curr - prev

    if diff > 0:
        if share_price:
            profit += curr - share_price.pop(0)
    elif diff < 0:
        share_price.append(curr)

    #print(curr, prev, diff, profit, len(share_price))
    #input()

    print(len(share_price), profit)


