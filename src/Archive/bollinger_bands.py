import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import pyotp
import robin_stocks.robinhood as robin_stocks

def login():
    with open(".api/.robinhood_api_key", "r") as file:
        content = file.read().split("\n")
        username = content[0]
        password = content[1]

    totp  = pyotp.TOTP("My2factorAppHere").now()
    login = robin_stocks.login(username,
                               password,
                               mfa_code=totp,
                               expiresIn=86400,)

login()

profile = robin_stocks.build_user_profile()

SEQUENCE_LEN = 20

tickers = [
    "AAPL",]#"MSFT","GOOGL","AMZN","TSLA","META","NVDA","ADBE","CRM","ORCL","INTC","IBM",
#    "QCOM","CSCO","ASML","TXN","AMD","SAP","SHOP","AVGO","INTU","SNOW","SQ","ZM","NFLX",  
#    "PYPL","GOOG","MS","V","MA","JPM","GS","WMT","TGT","HD","LOW","NKE","DIS",
#    "CMCSA","PEP","KO","T","VZ","AAP","F",
#]

tickers=["CHEF"]
#tickers=["BCH-USD"]

# Get current ask and bid prices for a stock
#quote = robin_stocks.stocks.get_stock_quote_by_symbol(tickers[0])
#quote = robin_stocks.crypto.get_crypto_quote(tickers[0].replace("-USD", ""))

#cur_price = float(quote["mark_price"])
#ask_price = float(quote['ask_price'])
#bid_price = float(quote['bid_price'])
#avg_price = (bid_price + ask_price) / 2
#dif_price = ask_price - bid_price

#print(f"Cur Price: {cur_price:.6f}, Avg Price: {avg_price:.6f}, Ask Price: {ask_price:.6f}, Bid Price: {bid_price:.6f}, Dif Price: {dif_price:.4f}")

def calculate_bollinger_bands(data, window=20, num_of_std=1):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def calculate_macd(data, slow=26, fast=12, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)
    # Simple Momentum Trading Logic
    # Buy signal: When MACD crosses above the signal line
    # Sell signal: When MACD crosses below the signal line
    ticker_df[ticker+'_trade_signal'] = 0  # Initialize column
    ticker_df.loc[macd > macd_signal, ticker+'_trade_signal'] = 'Buy'
    ticker_df.loc[macd < macd_signal, ticker+'_trade_signal'] = 'Sell'
"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


# List to hold data for each ticker
ticker_data_frames = []

for ticker in tickers:
    # Download historical data for the ticker
    #data = yf.download(ticker, start=start_date, end=end_date)
    #data = yf.download(ticker, period="max", interval="1d")
    #data = yf.download(ticker, period="60d", interval="90m")
    #data = yf.download(ticker, period="730d", interval="60m")
    #data = yf.download(ticker, period="60d", interval="30m")
    #data = yf.download(ticker, period="60d", interval="15m")
    #data = yf.download(ticker, period="60d", interval="5m")
    data = yf.download(ticker, period="max", interval="1m")
    print(data)


    # Calculate the daily percentage change
    close = data['Close']
    upper, lower = calculate_bollinger_bands(close)
    macd, signal = calculate_macd(close)
    
    percent_change_close = data['Close'].pct_change() * 100
    percent_upper, percent_lower = calculate_bollinger_bands(percent_change_close)

    # Create a DataFrame for the current ticker and append it to the list
    ticker_df = pd.DataFrame({
        ticker+'_close': close,
        ticker+'_upper': upper,
        ticker+'_lower': lower,
        ticker+'_macd': macd,
        ticker+'_signal' : signal,
        ticker+'_percent_close': percent_change_close,
        ticker+'_percent_upper': percent_upper,
        ticker+'_percent_lower': percent_lower,
        
    })
    ticker_data_frames.append(ticker_df)

# Concatenate all ticker DataFrames
percent_change_data = pd.concat(ticker_data_frames, axis=1)

# Remove any NaN values that may have occurred from the pct_change() calculation
percent_change_data.replace([np.inf, -np.inf], np.nan, inplace=True)
percent_change_data.dropna(inplace=True)

print(percent_change_data)

# Function to create X-day sequences for each ticker
def create_sequences(data, sequence_length=SEQUENCE_LEN):
    sequences = []
    data_size = len(data)
    for i in range(data_size - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# Example trading strategy for labeling
def define_label(percent_change):
    return percent_change

# Shift the percentage change data to create labels
labels = percent_change_data.shift(-1).map(define_label)

# Drop the last row in both percent_change_data and labels as it won't have a corresponding label
percent_change_data = percent_change_data.iloc[:-1]
labels = labels.iloc[:-1]


sequences_dict = {}
sequence_labels = {}
for ticker in tickers:
    # Extract close and volume data for the ticker
    close = percent_change_data[ticker+'_close'].values
    upper = percent_change_data[ticker+'_upper'].values
    lower = percent_change_data[ticker+'_lower'].values

    macd   = percent_change_data[ticker+'_macd'].values
    signal = percent_change_data[ticker+'_signal'].values

    percent_close = percent_change_data[ticker+'_percent_close'].values
    percent_upper = percent_change_data[ticker+'_percent_upper'].values
    percent_lower = percent_change_data[ticker+'_percent_lower'].values
    
    
    # Combine close and volume data
    ticker_data = np.column_stack((close,
                                   upper,
                                   lower,
                                   macd,
                                   signal,
                                   percent_close,
                                   percent_upper,
                                   percent_lower,))
    
    # Generate sequences
    ticker_sequences = create_sequences(ticker_data)
    sequences_dict[ticker] = ticker_sequences

    # Align labels with sequences
    sequence_labels[ticker] = labels[ticker+'_close'].values[SEQUENCE_LEN-1:]

# Combine data and labels from all tickers
all_sequences = []
all_labels = []

for ticker in tickers:
    all_sequences.extend(sequences_dict[ticker])
    all_labels.extend(sequence_labels[ticker])

# Convert to numpy arrays
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)


# Features for the Bollinger Band plot
features_bollinger = ['Close', 'Upper Band', 'Lower Band']
# Indices for MACD and Signal in the sequence
index_macd = 3  # Assuming 'MACD' is the fourth feature
index_signal = 4  # Assuming 'Signal' is the fifth feature

for i, (sequence, label) in enumerate(zip(all_sequences, all_labels)):
    if i == 5:
        break  # Stop after 5 sequences

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns

    # First subplot for Bollinger Bands and Close
    for j in range(3):  # For 'Close', 'Upper Band', and 'Lower Band'
        ax1.plot(sequence[:, j], label=f'{features_bollinger[j]}', marker='o')
    ax1.axhline(y=label, color='r', linestyle='-', label=f"Label: {label}")
    ax1.set_title(f"Bollinger Bands - Sequence {i+1}")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    ax1.legend()

    # Second subplot for MACD and Signal
    ax2.plot(sequence[:, index_macd], label='MACD', marker='o')
    ax2.plot(sequence[:, index_signal], label='Signal', marker='o')
    ax2.set_title(f"MACD - Sequence {i+1}")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("MACD Value")
    ax2.legend()

    # Save the combined figure
    plt.savefig(f"data/combined_{i:04d}.jpg")  # Save the image
    plt.close(fig)
exit()

starting_budget = budget = 100
num_shares = 1
own = 0
bought_at = []
profit_loss = 0
# Define a threshold for 'near'
threshold_percent = 0.025 # 1% for near

for i, (sequence, label) in enumerate(zip(all_sequences, all_labels)):

    # Extract last values
    last_close = sequence[-1, 0]  # Assuming 'Close' is the first column
    last_upper_band = sequence[-1, 1]  # Assuming 'Upper Band' is the second column
    last_lower_band = sequence[-1, 2]  # Assuming 'Lower Band' is the third column

    macd = sequence[-1, 3]
    signal = sequence[-1, 4]

    # Calculate 'near' thresholds
    upper_threshold = last_upper_band - last_upper_band * (threshold_percent / 100)
    lower_threshold = last_lower_band + last_lower_band * (threshold_percent / 100)

    # Check conditions
    if last_close >= upper_threshold:
    #if macd >= signal:
        for _ in range(0, num_shares):
            if own:
                own -= 1
                purchase_price = bought_at.pop(0)
                delta =  last_close - purchase_price
                pct_delta = (delta / purchase_price) * 100
                profit_loss += delta
                budget += last_close
                print(f"SELL {own=:.1f}, {delta=:.4f}, {pct_delta=:.4f}, {profit_loss=:.4f}, {budget=:.4f}")
    elif last_close <= lower_threshold:
    #elif macd <= signal:
        for _ in range(0, num_shares):
            if last_close < budget:
                
                own += 1
                bought_at.append(last_close)
                budget -= last_close
                print(f"PURCHASE {own=:.1f}, {profit_loss=:.4f}, {budget=:.4f}")
            else:
                print(f"Cannot purchase {last_close=:.4f} !<= {budget=:.4f}")
    #else:
    #    print("Last close is within the Bollinger Bands.")



while len(bought_at):
    own -= 1
    purchase_price = bought_at.pop(0)
    delta =  last_close - purchase_price
    pct_delta = (delta / purchase_price) * 100
    profit_loss += delta
    budget += last_close
    print(f"SELL OFF : {own=:.1f}, {delta=:.4f}, {pct_delta=:.4f}, {profit_loss=:.4f}, {budget=:.4f}")

    
first_close_price = all_sequences[0][0, 0]
last_close_price = all_sequences[-1][-1, 0]
num_shares = starting_budget // first_close_price
just_holding = (last_close_price - first_close_price) * num_shares

print(f"{just_holding=:.4f}")
