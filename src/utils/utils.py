import numpy as np
import pandas as pd
import yfinance as yf

def get_stock_data(symbol: str,
                   peroid: str,
                   interval: str) -> pd.DataFrame:
    # Fetch historical stock data using yfinance
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period=peroid, interval=interval)
    
    return stock_data

def make_time_series(data: pd.DataFrame,
                     num_in_sequence: int) -> np.array:
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_array = data[columns].values

    output = []
    label = []
    for i in range(len(data_array)-num_in_sequence):
        output.append(data_array[i:i+num_in_sequence])
        label.append([data_array[i+num_in_sequence, 3]])
    return np.array(output), np.array(label)
