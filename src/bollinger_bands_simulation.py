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

def get_quote(ticker, stock=False):
    if stock:
        return rh.stocks.get_quotes(ticker)[0]
    else :
        return rh.crypto.get_crypto_quote(ticker)

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
    data['open_price'] = pd.to_numeric(data['open_price'], errors='coerce')
    data['close_price'] = pd.to_numeric(data['close_price'], errors='coerce')
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))  # 1 row, 2 columns
    
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

#############################################################################################################

################
## PARAMETERS ##
################
SEQUENCE_LEN = 30
stock = False
#ticker="AVAX"
#ticker="XLM"
#ticker="ETH"
#ticker="BTC"

#stock = True
#ticker="AAPL"

###########
## LOGIN ##
###########
login()
profile = rh.build_user_profile()

##############
## GET DATA ##
##############
data = get_data(ticker, interval='15second', span='hour')
#data = get_data(ticker, interval='5minute', span='week', stock=stock)
print(data)

######################
## CREATE SEQUENCES ##
######################
sequences = create_sequences(data, sequence_len=SEQUENCE_LEN)

##########
## PLOT ##
##########
plot(sequences)


#################
## TRADE LOGIC ##
#################
starting_budget = budget = 1000
own = 0
bought_at = []
profit_loss = 0
num_shares = []

for i, sequence in enumerate(sequences):

    # Extract last values
    last_close = sequence['close_price'].iloc[-1]
    last_upper_band = sequence['upper'].iloc[-1]
    last_lower_band = sequence['lower'].iloc[-1]

    # Check conditions
    if last_close >= last_upper_band:
        
        if own:
            
            purchase_price = bought_at.pop(0)
            number_shares = num_shares.pop(0)

            own -= number_shares
            
            delta =  last_close - purchase_price
            pct_delta = (delta / purchase_price) * 100
            profit_loss += (delta * number_shares)
            budget += (last_close * number_shares)
            print(f"SELL {own=:.1f}, {delta=:.4f}, {pct_delta=:.4f}, {profit_loss=:.4f}, {budget=:.4f}, {number_shares=}")
    elif last_close <= last_lower_band:
        
        if last_close < budget:
            
            half = budget / 2 if budget / 2 > last_close else budget
            number_shares = half // last_close
            num_shares.append(number_shares)
            own += number_shares
            bought_at.append(last_close)
            budget -= (last_close * number_shares)
            print(f"PURCHASE {own=:.1f}, {profit_loss=:.4f}, {budget=:.4f}, {number_shares=}")
        else:
            print(f"Cannot purchase {last_close=:.4f} !<= {budget=:.4f}")


while len(bought_at):
    own -= 1
    purchase_price = bought_at.pop(0)
    number_shares = num_shares.pop(0)
    delta =  last_close - purchase_price
    pct_delta = (delta / purchase_price) * 100
    profit_loss += (delta * number_shares)
    budget += (last_close * number_shares)
    print(f"SELL OFF : {own=:.1f}, {delta=:.4f}, {pct_delta=:.4f}, {profit_loss=:.4f}, {budget=:.4f}, {number_shares}")

    
first_close_price = sequences[0]['close_price'].iloc[0]
last_close_price = sequences[-1]['close_price'].iloc[-1]
num_shares = starting_budget // first_close_price
just_holding = (last_close_price - first_close_price) * num_shares

print(f"{just_holding=:.4f}")

def simulation(ticker='XLM', SEQUENCE_LEN=20, stock=False):

    starting_budget = budget = 1000
    own = 0
    bought_at = []
    profit_loss = 0
    num_shares = []

    quote = get_quote(ticker, stock=stock)
    ask_price = float(quote['ask_price'])
    bid_price = float(quote['bid_price'])
    print(f"{ask_price=:.4f}, {bid_price=:0.4f}, {ask_price-bid_price=:.4f}, {((ask_price-bid_price)/ask_price)*100=:.4f}")
    first_close_price = ask_price

    while True:
        
        # Login
        login()
        profile = rh.build_user_profile()
    
        # Get data
        data = get_data(ticker, interval='15second', span='hour')
        #data = get_data(ticker, interval='5minute', span='week', stock=stock)

        # Create sequences
        sequences = create_sequences(data, sequence_len=SEQUENCE_LEN,)

        # Plot
        plot(sequences)

        #################
        ## Trade logic ##
        #################
        sequence = sequences[-1]

        # Extract last values
        last_close = sequence['close_price'].iloc[-1]
        last_upper_band = sequence['upper'].iloc[-1]
        last_lower_band = sequence['lower'].iloc[-1]

        # Get current ask and bid prices
        quote = get_quote(ticker, stock=stock)
        ask_price = float(quote['ask_price'])
        bid_price = float(quote['bid_price'])

        # Calculate hypothetical if we just bought
        last_close_price = bid_price
        number_shares = starting_budget // first_close_price
        just_holding = (last_close_price - first_close_price) * number_shares

        # Check conditions
        if last_close >= last_upper_band:
        
            if own:
            
                purchase_price = bought_at.pop(0)
                number_shares = num_shares.pop(0)

                own -= number_shares
            
                delta =  bid_price - purchase_price
                pct_delta = (delta / purchase_price) * 100
                profit_loss += (delta * number_shares)
                budget += (bid_price * number_shares)
                print(f"SELL {own=:.1f}, {delta=:.4f}, {pct_delta=:.4f}, {profit_loss=:.4f}, {budget=:.4f}, {number_shares=}, {just_holding=:.4f}")
                
        elif last_close <= last_lower_band:
        
            if ask_price < budget:
            
                half = budget / 2 if budget / 2 > ask_price else budget
                number_shares = half // ask_price
                num_shares.append(number_shares)
                
                own += number_shares
                
                bought_at.append(ask_price)
                budget -= (ask_price * number_shares)
                print(f"PURCHASE {own=:.1f}, {profit_loss=:.4f}, {budget=:.4f}, {number_shares=}, {just_holding=:.4f}")
                
            else:
                print(f"Cannot purchase {last_close=:.4f} !<= {budget=:.4f}")

        # Wait for 15 seconds before next iteration
        time.sleep(15)
    


simulation(ticker=ticker, SEQUENCE_LEN=SEQUENCE_LEN, stock=stock)
