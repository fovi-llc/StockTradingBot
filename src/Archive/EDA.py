import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf


tickers = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","ADBE","CRM","ORCL","INTC","IBM",
    "QCOM","CSCO","ASML","TXN","AMD","SAP","SHOP","AVGO","INTU","SNOW","SQ","ZM","NFLX",  
    "PYPL","GOOG","MS","V","MA","JPM","GS","WMT","TGT","HD","LOW","NKE","DIS",
    "CMCSA","PEP","KO","T","VZ","AAP","F",
]
#tickers = [
#    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "ADBE", "CRM", "ORCL", "INTC", "IBM",
#    "QCOM", "CSCO", "ASML", "TXN", "AMD", "SAP", "SHOP", "AVGO", "INTU", "SNOW", "SQ", "ZM", "NFLX",
#    "PYPL", "GOOG", "META", "MS", "V", "MA", "JPM", "GS", "WMT", "TGT", "HD", "LOW", "NKE", "DIS",
#    "CMCSA", "PEP", "KO", "T", "VZ", "AAP", "F", "BAC", "UBS", "XOM", "CVX", "PG", "GE", "MMM",
#    "BA", "CAT", "JNJ", "GILD", "ABBV", "MRK", "PFE", "BMY", "UNH", "MCD", "KO", "LMT", "GD",
#    "RTX", "DE", "GM", "FORD", "DAL", "AAL", "UAL", "LUV", "ALK", "SPG", "CBRE", "PLD", "CCI",
#    "SBUX", "YUM", "MGM", "WYNN", "BKNG", "EXPE", "MAR", "HLT", "LVS", "EBAY", "TWTR", "SNAP",
#    "ROKU", "ETSY", "TEAM", "DOCU", "ZS", "CRWD", "OKTA", "DDOG", "FTNT", "ANSS", "KEYS", "KLAC",
#    "LRCX", "MRVL", "MTCH", "MELI", "PDD", "SE", "BIDU", "JD", "BABA", "NTES", "TCEHY", "NVAX",
#    "MRNA", "BIIB", "CELG", "REGN", "VRTX", "ILMN", "ALGN", "SYK", "ISRG", "EW", "ABT", "A"
#]


# Calculate the start date
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=25*365)

def moving_average(data, window_size=2):
    """Apply a simple moving average to the data."""
    return data.rolling(window=window_size).mean()

# List to hold data for each ticker
ticker_data_frames = []

for ticker in tickers:
    # Download historical data for the ticker
    #data = yf.download(ticker, start=start_date, end=end_date)
    #data = yf.download(ticker, period="max", interval="1d")
    #data = yf.download(ticker, period="60d", interval="90m")
    data = yf.download(ticker, period="730d", interval="60m")
    #data = yf.download(ticker, period="60d", interval="30m")
    #data = yf.download(ticker, period="60d", interval="15m")
    #data = yf.download(ticker, period="60d", interval="5m")
    #data = yf.download(ticker, period="7d", interval="1m")
    
    # Calculate the daily percentage change
    percent_change_close = data['Close'].pct_change() * 100
    percent_change_volume = data['Volume'].pct_change() * 100
    
    # Apply smoothing (moving average)
    smoothed_close = moving_average(percent_change_close)
    smoothed_volume = moving_average(percent_change_volume)
    
    # Create a DataFrame for the current ticker and append it to the list
    ticker_df = pd.DataFrame({
        ticker+'_close': percent_change_close,
        ticker+'_volume': percent_change_volume,
        ticker+'_close_smooth': smoothed_close,
        ticker+'_volume_smooth': smoothed_volume
    })
    ticker_data_frames.append(ticker_df)

# Concatenate all ticker DataFrames
percent_change_data = pd.concat(ticker_data_frames, axis=1)

# Remove any NaN values that may have occurred from the pct_change() calculation
percent_change_data.replace([np.inf, -np.inf], np.nan, inplace=True)
percent_change_data.dropna(inplace=True)

print(percent_change_data[:15])

pattern_dict = {}
for num_days in range(3, 9):
    total = 0
    both_pos = 0
    for ticker in tickers:
        percent = np.array(percent_change_data[ticker+'_close'])
        
        for i in range(0, len(percent)-num_days+1):
            subset = percent[i:i+num_days]
            prev_days = subset[:-1]
            curr_day = subset[-1]
            
            # Define Pattern
            pattern = ""
            flag=False
            for num in prev_days:
                #if num > 0 : pattern += "+"
                #else: pattern += "-"

                num = round(num,10)
                if num > 0 : pattern += "+"
                elif num < 0 : pattern += "-"
                else: flag=True ; break
            if flag: continue
            
            # Init Dict
            if pattern not in pattern_dict:
                pattern_dict[pattern] = {"tot":0, "pos":0, "neg":0}

            # increment Dict
            pattern_dict[pattern]["tot"] += 1
            if curr_day > 0: pattern_dict[pattern]["pos"] += 1
            else: pattern_dict[pattern]["neg"] += 1
            
            
for pattern, values in pattern_dict.items():
    tot = values["tot"]
    pos = values["pos"]
    neg = values["neg"]

    pos_pct = (pos/tot)*100
    neg_pct = (neg/tot)*100

    pattern_dict[pattern]["pos_pct"] = round(pos_pct,2)
    pattern_dict[pattern]["neg_pct"] = round(neg_pct,2)
        
    print(f"{pattern=}, {tot=}, {pos=}, {neg=}, {pos_pct=:.2f}, {neg_pct=:.2f}")

# Sort the dictionary based on 'pos_pct' in descending order
sorted_patterns = sorted(pattern_dict.items(), key=lambda x: x[1]['pos_pct'], reverse=True)

# Now, sorted_patterns is a list of tuples, where each tuple is (pattern, dictionary of values)
# You can iterate over it and print or use as needed
for pattern, values in sorted_patterns:
    print(f"{pattern}: {values}")
