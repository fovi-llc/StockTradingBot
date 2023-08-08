import sys
import time
import random
import warnings
import select
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import pyotp
import robin_stocks.robinhood as robin_stocks

# Colors
black = '\033[30m'
red = '\033[31m'
green = '\033[32m'
yellow = '\033[33m'
blue = '\033[34m'
magenta = '\033[35m'
cyan = '\033[36m'
gray = '\033[90m'
light_red = '\033[91m'
light_green = '\033[92m'
light_yellow = '\033[93m'
light_blue = '\033[94m'
light_magenta = '\033[95m'
light_cyan = '\033[96m'
light_gray = '\033[37m'
white = '\033[37m'
reset = '\033[0m'

def sell_crypto(symbol):
    # get latest prices
    quote = robin_stocks.crypto.get_crypto_quote(symbol)
                    
    # Asking Price: The asking price is the lowest price at which a seller is willing to sell
    ask_price = float(quote["ask_price"])
    bid_price = float(quote["bid_price"])
    round_price = round((ask_price + bid_price) / 2, 6)

    # place order
    order = robin_stocks.orders.order_sell_crypto_limit(symbol, quantity=1, limitPrice=round_price)

    print(f"{light_blue}PLACE SELL ORDER @ {round_price=:.6f}{reset}")
    
    return order["id"]

def buy_crypto(symbol):
    
    # get latest prices
    quote = robin_stocks.crypto.get_crypto_quote(symbol)
                    
    # Bid Price: The bid price is the highest price that a buyer is willing to pay
    ask_price = float(quote["ask_price"])
    bid_price = float(quote["bid_price"])
    round_price = round((ask_price + bid_price) / 2, 6)

    # place order
    order = robin_stocks.orders.order_buy_crypto_limit(symbol, quantity=1, limitPrice=round_price)

    print(f"{light_green}PLACE BUY ORDER @ {round_price=:.6f}{reset}")

    return order["id"]

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

def get_user_input(symbol='DOGE',
                   interval='15second',
                   span='hour',
                   UNITS=1,
                   SELL_ALL=False,
                   METRIC='prices',
                   plot_metrics=True,
                   PURCHASE=True):
    total_time = 15

    start_time = time.time()
    while True:
        time_left = total_time - (time.time() - start_time)
        if time_left <= 0:
            break

        print(f"  {time_left:.0f}/{total_time:.0f} (Y/N) to enter input mode ", end="\r")

        i, o, e = select.select([sys.stdin], [], [], 1)  # Check for input every 1 second

        if i:
            user_input = sys.stdin.readline().strip().upper()
            if user_input == "Y":
                process_user_input(user_input)
                print("#################################")
                print("## HELLO WELCOME TO INPUT MODE ##")
                print("#################################")
                print()
                print("######################")
                print("## CURRENT SETTINGS ##")
                print("######################")
                print(f"Press '0' to modify {symbol=}")
                print(f"Press '1' to modify {interval=}")
                print(f"Press '2' to modify {span=}")
                print(f"Press '3' to modify {UNITS=}")
                print(f"Press '4' to modify {SELL_ALL=}")
                print(f"Press '5' to modify {METRIC=}")
                print(f"Press '6' to modify {plot_metrics=}")
                print(f"Press '7' to modify {PURCHASE=}")
                user_input = input().strip()
                if user_input == "0":
                    print(f"What would you like to change {symbol=} to?")
                    symbol = input().strip()
                    print(f"UPDATE: {symbol=}")
                    
                elif user_input == "1":
                    print(f"What would you like to change {interval=} to?")
                    interval = input().strip()
                    print(f"UPDATE: {interval=}")
                    
                elif user_input == "2":
                    print(f"What would you like to change {span=} to?")
                    span = input().strip()
                    print(f"UPDATE: {span=}")
                    
                elif user_input == "3":
                    print(f"What would you like to change {UNITS=} to?")
                    UNITS = int(input().strip())
                    print(f"UPDATE: {UNITS=}")
                    
                elif user_input == "4":
                    print(f"What would you like to change {SELL_ALL=} to?")
                    user_input = input().strip()
                    if user_input == "False": SELL_ALL = False
                    else: SELL_ALL = True
                    print(f"UPDATE: {SELL_ALL=}")
                    
                elif user_input == "5":
                    print(f"What would you like to change {METRIC=} to?")
                    METRIC = input().strip()
                    print(f"UPDATE: {METRIC=}")

                elif user_input == "6":
                    print(f"What would you like to change {plot_metrics=} to?")
                    user_input = input().strip()
                    if user_input == "False": plot_metrics = False
                    else: plot_metrics = True
                    print(f"UPDATE: {plot_metrics=}")
                elif user_input == "7":
                    print(f"What would you like to change {PURCHASE=} to?")
                    user_input = input().strip()
                    if user_input == "False": PURCHASE = False
                    else: PURCHASE = True
                    print(f"UPDATE: {PURCHASE=}")
                break
    return symbol, interval, span, UNITS, SELL_ALL, METRIC, plot_metrics, PURCHASE

def process_user_input(input_str):
    if input_str:
        print(f"User input: {input_str}")

def get_crypto_data(symbol="DOGE", interval='15second', span='hour'):
    data = robin_stocks.crypto.get_crypto_historicals(symbol=symbol,
                                                      interval=interval,
                                                      span=span,
                                                      bounds='24_7',
                                                      info=None)
    data = pd.DataFrame(data)
    data.rename(columns={
        'begins_at': 'Datetime',
        'open_price': 'Open',
        'close_price': 'Close',
        'high_price': 'High',
        'low_price': 'Low',
        'volume': 'Volume',
        'session': 'Session',
        'interpolated': 'Interpolated',
        'symbol': 'Symbol'
    }, inplace=True)

    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.sort_values(by='Datetime', inplace=True)

    # Reset the index (optional but recommended)
    data.reset_index(drop=True, inplace=True)

    data["Open"] = np.array(data["Open"], dtype=np.float64)
    data["Close"] = np.array(data["Close"], dtype=np.float64)
    data["High"] = np.array(data["High"], dtype=np.float64)
    data["Low"] = np.array(data["Low"], dtype=np.float64)
    data["Volume"] = np.array(data["Volume"], dtype=np.float64)

    return data

def get_stock_data(symbol: str,
                   peroid: str,
                   interval: str) -> pd.DataFrame:
    # Fetch historical stock data using yfinance
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period=peroid, interval=interval)
    
    return stock_data

def make_time_series(data: pd.DataFrame,
                     num_in_sequence: int) -> np.array:
    columns = ['Open', 'High', 'Low', 'Close']
    data_array = data[columns].values

    output = []
    label = []
    for i in range(len(data_array)-num_in_sequence):
        output.append(data_array[i:i+num_in_sequence])
        label.append([data_array[i+num_in_sequence, 3]])

    output = np.array(output, dtype=np.float32)
    label = np.array(label, dtype=np.float32)

    return tf.convert_to_tensor(output), tf.convert_to_tensor(label)

def make_prediction_series(data: pd.DataFrame,
                           num_in_sequence: int) -> np.array:
    columns = ['Open', 'High', 'Low', 'Close']
    data_array = data[columns].values

    output = []
    label = []
    for i in range(len(data_array)-num_in_sequence):
        output.append(data_array[i+1:i+num_in_sequence+1])
        label.append([data_array[i+num_in_sequence, 3]])
    
    return np.array(output), np.array(label)

def create_batches(data, labels, batch_size):
    num_samples = data.shape[0]
    num_batches = num_samples // batch_size
    
    # Compute the number of samples in the last batch
    remaining_samples = num_samples % batch_size
    
    # Split the data and labels into batches
    data_batches = np.split(data[:num_batches * batch_size], num_batches)
    label_batches = np.split(labels[:num_batches * batch_size], num_batches)
    
    # Add the remaining samples as a separate batch
    if remaining_samples > 0:
        data_batches.append(tf.convert_to_tensor(data[num_batches * batch_size:]))
        label_batches.append(tf.convert_to_tensor(labels[num_batches * batch_size:]))

    # convert_to_tensor
    data_batches = [tf.convert_to_tensor(batch) for batch in data_batches]
    label_batches = [tf.convert_to_tensor(batch) for batch in label_batches]

    # shuffle
    data_batches, label_batches = shuffle(data_batches, label_batches)
    
    return data_batches, label_batches

def shuffle(data, labels):
    zipped_data = list(zip(data, labels))
    random.shuffle(zipped_data)
    shuffled_data, shuffled_labels = zip(*zipped_data)
    return list(shuffled_data), list(shuffled_labels)
