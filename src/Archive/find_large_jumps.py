import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pyotp
import robin_stocks.robinhood as rh

def login():
    with open(".api/.robinhood_api_key", "r") as file:
        content = file.read().split("\n")
        username = content[0]
        password = content[1]

    totp  = pyotp.TOTP("My2factorAppHere").now()
    login = rh.login(username,
                               password,
                               mfa_code=totp,
                               expiresIn=86400,)

def calculate_bollinger_bands(data, window=20, num_of_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def calculate_macd(data, slow=26, fast=12, signal=9):
    """
    Calculate Moving Average Convergence Divergence (MACD)
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

def get_data(ticker, interval='5minute', span='week', stock=False):
    """
    Fetches historical data for a given ticker symbol from Robinhood. 

    Parameters:
    - ticker (str): The ticker symbol for the cryptocurrency or stock.
    - interval (str): The interval for the historical data. Default is '5minuet' for
                      cryptocurrencies. For stocks, the default and typical interval
                      is larger, like '5minute'.
    - span (str): The time span for the historical data. Default is 'week'.

    Returns:
    - pandas.DataFrame: A DataFrame containing the historical data along with calculated
                         indicators like Bollinger Bands and MACD. The DataFrame is cleaned
                         of any NaN values and infinite values.
    """

    if not stock:
        historical_data = rh.crypto.get_crypto_historicals(ticker, interval=interval, span=span)
    else:
        historical_data = rh.stocks.get_stock_historicals(ticker, interval=interval, span=span)

    data = pd.DataFrame(historical_data)

    # Convert each specified column to float
    data['begins_at'] = pd.to_datetime(data['begins_at'])
    data['begins_at'] = data['begins_at'].dt.tz_convert('America/New_York')
    data['open_price'] = pd.to_numeric(data['open_price'], errors='coerce')
    data['close_price'] = pd.to_numeric(data['close_price'], errors='coerce')
    data['close_price'] = data['close_price'].pct_change() * 100
    data['high_price'] = pd.to_numeric(data['high_price'], errors='coerce')
    data['low_price'] = pd.to_numeric(data['low_price'], errors='coerce')

    # Calculate the daily percentage change
    upper, lower = calculate_bollinger_bands(data['close_price'])
    macd, signal = calculate_macd(data['close_price'])

    data['upper'] = upper
    data['lower'] = lower
    data['macd'] = macd
    data['signal'] = signal

    
    
    # Remove any NaN values that may have occurred from the pct_change() calculation
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    return data

def create_sequences(data, sequence_len=20):
    """
    Creates overlapping sequences of a specified length from the provided DataFrame.

    Parameters:
    - data (pandas.DataFrame): The DataFrame from which to create sequences.
    - sequence_len (int): The length of each sequence. Default is 20.

    Returns:
    - list: A list of pandas DataFrame objects, each containing a sequence of data
            of length 'sequence_len'.
    """
    sequences = []
    for i in range(len(data) - sequence_len + 1):
        sequence = data.iloc[i:i + sequence_len]

        # Get start day and final day
        start_day = str(sequence['begins_at'].iloc[0]).split(' ')[0]
        final_day = str(sequence['begins_at'].iloc[-1]).split(' ')[0]

        # Only keep data if sequence is in the same day
        if start_day == final_day:
            sequences.append(sequence)
    return sequences

def plot(sequences):
    """
    Plots the last five sequences in a given list of sequences.

    Parameters:
    - sequences (list of pandas.DataFrame): A list of DataFrame objects, each containing
      a sequence of financial data.
    """
    for i, sequence in enumerate(sequences[-5:]):
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))  # 1 row, 2 columns
    
        # First subplot for Open, High, Low, Close, Upper, Lower
        ax1.plot(sequence['begins_at'], sequence['close_price'], label='Close Price', marker='o',)
        ax1.plot(sequence['begins_at'], sequence['upper'], label='Upper Band', marker='o')
        ax1.plot(sequence['begins_at'], sequence['lower'], label='Lower Band', marker='o')
        ax1.set_title(f'Price and Bollinger Bands - Sequence {i+1}')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Price')
        ax1.legend()
    
        # Second subplot for MACD, Signal
        ax2.plot(sequence['begins_at'], sequence['macd'], label='MACD', marker='o')
        ax2.plot(sequence['begins_at'], sequence['signal'], label='Signal Line', marker='o')
        ax2.set_title(f'MACD and Signal - Sequence {i+1}')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Value')
        ax2.legend()
    
        # Set x-ticks to each data point
        ax1.set_xticks(sequence['begins_at'])
        ax2.set_xticks(sequence['begins_at'])
        
        # Format x-axis to display time in HH:MM:SS AM/PM format
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M:%S %p'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M:%S %p'))
    
        # Rotate date labels for better readability
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    
        # Save the combined figure
        plt.savefig(f"data/sequence_plot_{i:04d}.jpg")  # Save the image
        plt.close(fig)

def find_large_changes(sequences, look_back_at=-4, threshold=0.20):
    large_increase_sequences = []
    large_decrease_sequences = []
    flat_sequences = []
    increase_diff = []
    decrease_diff = []
    flat_diff = []
    for sequence in sequences:


        start_price = sequence["close_price"].iloc[look_back_at] + 10e-10
        final_price = sequence["close_price"].iloc[-1]

        difference = final_price - start_price

        if difference > threshold:
            large_increase_sequences.append(sequence)
            increase_diff.append(difference)
        elif difference < threshold*(-1):
            large_decrease_sequences.append(sequence)
            decrease_diff.append(difference)
        elif difference < (threshold/2) and difference > (threshold/2)*(-1):
            flat_sequences.append(sequence)
            flat_diff.append(difference)
                
    return large_increase_sequences, large_decrease_sequences, flat_sequences

def simple_plot(sequences, name="sequence"):
   
    for i, sequence in enumerate(sequences):
        ticker = sequence['symbol'].iloc[0]
        
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(sequence['begins_at'], sequence['close_price'], label='Close Price', marker='o',)
        ax.plot(sequence['begins_at'], sequence['upper'], label='Upper Band', marker='o')
        ax.plot(sequence['begins_at'], sequence['lower'], label='Lower Band', marker='o')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price')
        ax.legend()
    
        # Set x-ticks to each data point
        ax.set_xticks(sequence['begins_at'])
    
        # Format x-axis to display time in HH:MM:SS AM/PM format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M:%S %p'))
        
        # Rotate date labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        
        # Saving the plot
        plt.savefig(f"data/{ticker}_{name}_{i:04d}.jpg")
        plt.close()
        
#############################################################################################################

################
## PARAMETERS ##
################
SEQUENCE_LEN = 40
ticker="AVAX"

tickers = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","ADBE","CRM","ORCL","INTC","IBM",
    "QCOM","CSCO","ASML","TXN","AMD","SAP","SHOP","AVGO","INTU","SNOW","SQ","ZM","NFLX",  
    "PYPL","GOOG","MS","V","MA","JPM","GS","WMT","TGT","HD","LOW","NKE","DIS",
    "CMCSA","PEP","KO","T","VZ","AAP","F",
]

############
## SET UP ##
############
if not os.path.exists("data/"): os.makedirs("data/")
###########
## LOGIN ##
###########
login()
profile = rh.build_user_profile()

##############
## GET DATA ##
##############
#data = get_data(ticker, interval='15second', span='hour')
data = get_data("AAPL", interval='5minute', span='week', stock=True)
print(data)

######################
## CREATE SEQUENCES ##
######################
sequences = create_sequences(data, sequence_len=SEQUENCE_LEN)
print(f"{len(sequences)} sequences")

##########
## PLOT ##
##########
plot(sequences)

#########################
## INCREASE & DECREASE ##
#########################
for ticker in tickers:
    login()
    profile = rh.build_user_profile()

    data = get_data(ticker, interval='5minute', span='week', stock=True)
    sequences = create_sequences(data, sequence_len=SEQUENCE_LEN)

    increase_sequences, decrease_sequences, flat_sequences  = find_large_changes(sequences, look_back_at=-10, threshold=0.20)

    print(f"{ticker} : {len(increase_sequences)=} {len(decrease_sequences)=}, {len(flat_sequences)=}")

    simple_plot(increase_sequences, name="increase")
    simple_plot(decrease_sequences, name="decrease")

    break
