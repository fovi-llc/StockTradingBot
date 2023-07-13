import random
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
    #columns = ['Open', 'High', 'Low', 'Close', 'Volume']
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


def train_val_test_split(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random=True):
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of train_ratio, val_ratio, and test_ratio must be 1.0"

    num_samples = data.shape[0]
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val

    # Split the data and labels into train, val, and test sets
    train_data = data[:num_train]
    train_labels = labels[:num_train]
    val_data = data[num_train:num_train + num_val]
    val_labels = labels[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]
    test_labels = labels[num_train + num_val:]

    if random:
        shuffled_indices = np.random.permutation(len(train_data))
        train_data, train_labels = train_data[shuffled_indices], train_labels[shuffled_indices]

        shuffled_indices = np.random.permutation(len(val_data))
        val_data, val_labels = val_data[shuffled_indices], val_labels[shuffled_indices]

        shuffled_indices = np.random.permutation(len(test_data))
        test_data, test_labels = test_data[shuffled_indices], test_labels[shuffled_indices]
    

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

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
        data_batches.append(data[num_batches * batch_size:])
        label_batches.append(labels[num_batches * batch_size:])

    return np.array(data_batches, dtype="object"), np.array(label_batches, dtype="object")

def shuffle(data, labels):
    zipped_data = list(zip(data, labels))
    random.shuffle(zipped_data)
    return zipped_data
