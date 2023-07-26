import sys
import time
import select
import threading

import requests
import pandas as pd

from utils import utils, NN, upward_trend, resistance_line

import yfinance as yf

def get_user_input():
    total_time = 15

    start_time = time.time()
    while True:
        time_left = total_time - (time.time() - start_time)
        if time_left <= 0:
            break

        print(f"  {time_left:.0f}/{total_time:.0f} (Y/N) to enter input mode ", end="\r")

        i, o, e = select.select([sys.stdin], [], [], 1)  # Check for input every 1 second

        if i:
            user_input = sys.stdin.readline().strip()
            process_user_input(user_input)
            return

def process_user_input(input_str):
    if input_str:
        print(f"User input: {input_str}")

def get_crypto_history_data(crypto_symbol, currency_symbol, limit=1000):
    base_url = 'https://min-api.cryptocompare.com/data/v2'
    endpoint = '/histominute'
    url = f'{base_url}{endpoint}?fsym={crypto_symbol}&tsym={currency_symbol}&limit={limit}'

    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data['Data']['Data'])
        df['Datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('Datetime', inplace=True)

        # Rename the columns to match yfinance column names
        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volumeto': 'Volume'}, inplace=True)

        # Drop any unnecessary columns
        df.drop(columns=['time', 'conversionType', 'conversionSymbol'], inplace=True)
        
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None

plot_metrics = True
def main():
    # Replace 'BTC' and 'USD' with the desired cryptocurrency and currency symbols
    crypto_symbol = 'ADA' # 'BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'XLM', 
    currency_symbol = 'USD'
    limit = 60  # Number of 1-minute data points to retrieve

    start_price = get_crypto_history_data(crypto_symbol, currency_symbol, limit)['Close'][-1]
    
    BUY_PRICES = []
    TOTAL = 0.0
    CASH = 10000
    
    while True:
        data = get_crypto_history_data(crypto_symbol, currency_symbol, limit)

        if plot_metrics:
            resistance_line.plot_metrics_with_resistance(crypto_symbol, data)
        prices, volume, RSIs, macd, bollinger_bands = resistance_line.above_or_below_resistance_line(crypto_symbol, data)

        RESISTANCE_LINE = RSIs
        current_price = data['Close'][-1]
        print(f"{RESISTANCE_LINE[-10:]=}")

        #################
        ## Trade Logic ##
        #################
        if RESISTANCE_LINE[-1] == "Below":
            
            if (CASH - current_price) > 0:
            
                BUY_PRICES.append(current_price)
                CASH -= current_price
    
                print(f"{utils.light_green}BUY @ {current_price=:.4f} with {CASH=:.4f} left{utils.reset}")
                
            else: print(f"{RESISTANCE_LINE[-1]=} cannot buy b/c {(CASH - current_price) > 0=:.4f}")
    
        elif RESISTANCE_LINE[-1] == "Above":
            if len(BUY_PRICES) > 0:
    
                TOTAL += current_price - BUY_PRICES[0]
                CASH += current_price
    
                print(f"{utils.light_blue}SELL @ {current_price=:.4f} (+-) {current_price - BUY_PRICES[0]=:.4f} {CASH=:.4f}{utils.reset}")
                BUY_PRICES = BUY_PRICES[1:]
                
            else: print(f"{utils.gray}{RESISTANCE_LINE[-1]=} cannot sell b/c {len(BUY_PRICES) > 0=}{utils.reset}")
    
        else: print("THIS SHOULD NEVER HAPPEN")
    
        ##############
        ## OVERVIEW ##
        ##############
        trade_pct = (((start_price + TOTAL)-start_price) / abs(start_price))* 100
        pct = (((current_price-start_price) / abs(start_price))* 100)
        if TOTAL < 0:
            print(f"{utils.light_red}{CASH=:.4f} {len(BUY_PRICES)=} {TOTAL=:.4f} {trade_pct=:.4f}%{utils.reset}")
        else:
            print(f"{utils.green}{CASH=:.4f} {len(BUY_PRICES)=} {TOTAL=:.4f} {trade_pct=:.4f}%{utils.reset}")
    
        if current_price - start_price < 0:
            print(f"Started trading @ {start_price=:.4f} Without trading bot {utils.light_red} {current_price - start_price=:.4f} {pct=:.4f}%{utils.reset}")
        else:
            print(f"Started trading @ {start_price=:.4f} Without trading bot {utils.light_yellow} {current_price - start_price=:.4f} {pct=:.4f}%{utils.reset}")
        print()
        
        #start_time = time.time()
        #while time.time() - start_time < 15:
        #    print(f"Run_Time {time.time() - start_time:.1f}", end="\r")
        #    time.sleep(1)
        get_user_input()
        
if __name__ == "__main__":
    main()
