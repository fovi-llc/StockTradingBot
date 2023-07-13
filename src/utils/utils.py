import random
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf

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
