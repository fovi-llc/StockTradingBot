import tqdm
from pathlib import Path
import datetime
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import BatchNormalization, Activation
import yfinance as yf
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Conv2D, TimeDistributed, Flatten, Reshape
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from utils import utils, NN, upward_trend, resistance_line

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.metrics import MeanAbsoluteError, BinaryAccuracy
import keras.backend as K

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


tickers = [
    "AAPL",]#"MSFT","GOOGL","AMZN","TSLA","META","NVDA","ADBE","CRM","ORCL","INTC","IBM",]
#    "QCOM","CSCO","ASML","TXN","AMD","SAP","SHOP","AVGO","INTU","SNOW","SQ","ZM","NFLX",  
#    "PYPL","GOOG","MS","V","MA","JPM","GS","WMT","TGT","HD","LOW","NKE","DIS",
#    "CMCSA","PEP","KO","T","VZ","AAP","F",
    #]

# Calculate the start date
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=25*365)

#########################
## Normalizing Methods ##
#########################
def min_max_scaling(series):
    """Apply Min-Max scaling to a Pandas Series."""
    return (series - series.min()) / (series.max() - series.min())

def scale_to_neg1_pos1(series):
    """Scale the data to a range of [-1, 1]."""
    return 2 * ((series - series.min()) / (series.max() - series.min())) - 1

def standardize(series):
    """Apply Standardization (Z-score normalization) to a Pandas Series."""
    return (series - series.mean()) / series.std()
#########################

def moving_average(data, window_size=2):
    """Apply a simple moving average to the data."""
    return data.rolling(window=window_size).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_bollinger_bands(data, window=20, num_of_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def calculate_stochastic_oscillator(data, window=14):
    """Calculate Stochastic Oscillator"""
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    stoch = 100 * (data['Close'] - low_min) / (high_max - low_min)
    return stoch

def calculate_historical_volatility(data, window=20):
    """Calculate Historical Volatility."""
    return data.pct_change().rolling(window=window).std() * np.sqrt(window)

def calculate_atr(data, window=14):
    """Calculate Average True Range (ATR)."""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_momentum(data, period=12):
    """Calculate Momentum."""
    return data['Close'] - data['Close'].shift(period)

def calculate_ema(data, window=20):
    """Calculate Exponential Moving Average (EMA)."""
    return data['Close'].ewm(span=window, adjust=False).mean()

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
    #percent_change_close = data['Close'].pct_change() * 100
    #percent_change_volume = scale_to_neg1_pos1(data['Volume'].pct_change() * 100)
    #
    ## Apply smoothing (moving average)
    #smoothed_close = moving_average(percent_change_close)
    #smoothed_volume = moving_average(percent_change_volume)
    #
    ## Calculate metrics
    #rsi               = scale_to_neg1_pos1(calculate_rsi(data['Close']))
    #macd, macd_signal = calculate_macd(data['Close'])
    #macd, macd_signal = scale_to_neg1_pos1(macd), scale_to_neg1_pos1(macd_signal)
    #upper, lower      = calculate_bollinger_bands(data['Close'])
    #upper, lower      = scale_to_neg1_pos1(upper), scale_to_neg1_pos1(lower)
    #stoch_oscill      = scale_to_neg1_pos1(calculate_stochastic_oscillator(data))
    #hist_volat        = scale_to_neg1_pos1(calculate_historical_volatility(data['Close']))
    #atr               = scale_to_neg1_pos1(calculate_atr(data))
    #momentum          = scale_to_neg1_pos1(calculate_momentum(data))
    #ema               = scale_to_neg1_pos1(calculate_ema(data))

    # Calculate the daily percentage change
    percent_change_close = data['Close'].pct_change() * 100
    percent_change_volume = data['Volume'].pct_change() * 100
    
    # Apply smoothing (moving average)
    smoothed_close = moving_average(percent_change_close)
    smoothed_volume = moving_average(percent_change_volume)

    # Calculate metrics
    rsi               = calculate_rsi(data['Close'])
    macd, macd_signal = calculate_macd(data['Close'])
    upper, lower      = calculate_bollinger_bands(data['Close'])
    stoch_oscill      = calculate_stochastic_oscillator(data)
    hist_volat        = calculate_historical_volatility(data['Close'])
    atr               = calculate_atr(data)
    momentum          = calculate_momentum(data)
    ema               = calculate_ema(data)
    
    # Create a DataFrame for the current ticker and append it to the list
    ticker_df = pd.DataFrame({
        ticker+'_close': percent_change_close,
        ticker+'_volume': percent_change_volume,
        ticker+'_close_smooth': smoothed_close,
        ticker+'_volume_smooth': smoothed_volume,
        ticker+'rsi': rsi,
        ticker+'macd': macd,
        ticker+'macd_signal': macd_signal,
        ticker+'upper': upper,
        ticker+'lower': lower,
        ticker+'stoch_oscill': stoch_oscill,
        ticker+'hist_volat': hist_volat,
        ticker+'atr': atr,
        ticker+'momentum': momentum,
        ticker+'ema': ema
    })
    ticker_data_frames.append(ticker_df)

# Concatenate all ticker DataFrames
percent_change_data = pd.concat(ticker_data_frames, axis=1)

# Remove any NaN values that may have occurred from the pct_change() calculation
percent_change_data.replace([np.inf, -np.inf], np.nan, inplace=True)
percent_change_data.dropna(inplace=True)

print(percent_change_data[:15])

total = 0
correct = 0
for ticker in tickers:
    
    for close, close_smooth in zip(percent_change_data[ticker+'_close'], percent_change_data[ticker+'_close_smooth']):
        total += 1
        if close < 0 and close_smooth < 0:
            correct += 1
        elif close > 0 and close_smooth >0:
            correct += 1

print(f"{correct/total=}")


# Function to create 30-day sequences for each ticker
def create_sequences(data, sequence_length=20):
    sequences = []
    data_size = len(data)
    for i in range(data_size - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# Example trading strategy for labeling
def define_label(percent_change):
    return percent_change
    #if percent_change > .00:  # If the stock increases by more than 1%
    #    return 'buy' #, percent_change, 0
    #elif percent_change <= -.00:  # If the stock decreases by more than 1%
    #    return 'sell' #, percent_change, 1
    #else:
    #    return 'hold' #, percent_change, 2

# Shift the percentage change data to create labels
labels = percent_change_data.shift(-1).map(define_label)

# Drop the last row in both percent_change_data and labels as it won't have a corresponding label
percent_change_data = percent_change_data.iloc[:-1]
labels = labels.iloc[:-1]

print(percent_change_data)
print(labels)

# Creating sequences and labels for each ticker
sequences_dict = {}
sequence_labels = {}
for ticker in tickers:
    # Extract close and volume data for the ticker
    close = percent_change_data[ticker+'_close'].values
    volume = percent_change_data[ticker+'_volume'].values
    close_smooth = percent_change_data[ticker+'_close_smooth'].values
    volume_smooth = percent_change_data[ticker+'_volume_smooth'].values
    rsi = percent_change_data[ticker+'rsi'].values
    macd = percent_change_data[ticker+'macd'].values
    macd_signal = percent_change_data[ticker+'macd_signal'].values
    upper = percent_change_data[ticker+'upper'].values
    lower = percent_change_data[ticker+'lower'].values
    stoch_oscill = percent_change_data[ticker+'stoch_oscill'].values
    hist_volat = percent_change_data[ticker+'hist_volat'].values
    atr = percent_change_data[ticker+'atr'].values
    momentum = percent_change_data[ticker+'momentum'].values
    ema = percent_change_data[ticker+'ema'].values
    
    # Combine close and volume data
    ticker_data = np.column_stack((close,
                                   volume,
                                   rsi,
                                   macd,
                                   macd_signal,
                                   upper,
                                   lower,
                                   stoch_oscill,
                                   hist_volat,
                                   atr,
                                   momentum,
                                   ema))
    #ticker_data = ticker_close_data
    
    # Generate sequences
    ticker_sequences = create_sequences(ticker_data)
    sequences_dict[ticker] = ticker_sequences

    # Align labels with sequences
    labels_close = labels[ticker+'_close'].values[19:]
    labels_smooth = labels[ticker+'_close_smooth'].values[19:]

    sequence_labels[ticker] = labels_close

# Combine data and labels from all tickers
all_sequences = []
all_labels = []

for ticker in tickers:
    all_sequences.extend(sequences_dict[ticker])
    all_labels.extend(sequence_labels[ticker])

# Convert to numpy arrays
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

# Print Data
# Define feature names for labeling
feature_names = ['Close',
                 'Volume',
                 'RSI',
                 'MACD',
                 'MACD Signal',
                 'Upper Band',
                 'Lower Band',
                 'Stochastic Oscillator',
                 'Historical Volatility',
                 'ATR',
                 'Momentum',
                 'ema',]

for i, (sequence, label) in enumerate(zip(all_sequences, all_labels)):
    plt.figure(figsize=(10, 6))
    
    # Plot each feature in the sequence
    for j in range(sequence.shape[1]):  # Assuming sequence.shape[1] is the number of features
        plt.plot(sequence[:, j], label=f'{feature_names[j]}', marker='o')  # Plot with line and markers

    plt.axhline(y=label, color='r', linestyle='-', label=f"Label: {label}")

    plt.title(f"Sequence {i+1} with Label {label}")
    plt.xlabel("Time Step")
    plt.ylabel("% Change")
    plt.legend()

    plt.savefig(f"data/{i:04d}.jpg")  # Ensure the "data/" directory exists
    plt.close()
    if i == 50: break
    #input()

#print(f"{np.average(all_sequences)=},{np.average(np.abs(all_sequences))=}")

# Encoding categorical labels
#label_encoder = LabelEncoder()
#all_labels = label_encoder.fit_transform(all_labels)

################################################################################
# Reshape the data to [num_samples, X, Y]
all_sequences = all_sequences.reshape(-1, 20, 12)
print(all_sequences.shape)
################################################################################
print("Total sequences:", len(all_sequences))
print("Total labels:", len(all_labels))

# Count occurrences of each class in the labels
label_counts = Counter(all_labels)

#print("Label distribution:")
#for label, count in label_counts.items():
#    print(f"{label}: {count}")

# Splitting the dataset into train, validation, and test sets
train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
    all_sequences, all_labels, train_size=0.90, random_state=42, shuffle=True)

validation_sequences, test_sequences, validation_labels, test_labels = train_test_split(
    temp_sequences, temp_labels, train_size=0.5, random_state=42, shuffle=True)

# Output sizes of each set
print("Training set size:", train_sequences.shape, train_labels.shape)
print("Validation set size:", validation_sequences.shape, validation_labels.shape)
print("Testing set size:", test_sequences.shape, test_labels.shape)
print(f"{np.average(test_labels)=},{np.average(np.abs(test_labels))=}")


# Neural Network Parameters
dropout_rate=0.5
regularizer_rate=1e-9

# Neural Network Model
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(20, 12), kernel_regularizer=l2(regularizer_rate)))
model.add(BatchNormalization())  # Batch Normalization after LSTM layer
model.add(Activation('leaky_relu'))
model.add(Dropout(dropout_rate))

model.add(LSTM(50, kernel_regularizer=l2(regularizer_rate)))
model.add(BatchNormalization())  # Batch Normalization after LSTM layer
model.add(Activation('leaky_relu')) 
model.add(Dropout(dropout_rate))

model.add(Dense(50, kernel_regularizer=l2(regularizer_rate)))
model.add(BatchNormalization())  # Batch Normalization before activation
model.add(Activation('leaky_relu')) 
model.add(Dropout(dropout_rate))

#model.add(Dense(len(label_encoder.classes_)))  # Output layer
model.add(Dense(units=1))

#model.load_weights("best_train_model.h5")

# Define a custom RMSE metric
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),]
#model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

loss_function = tf.keras.losses.MeanSquaredError(name="MSE")
metrics = [ MeanAbsoluteError(), rmse]
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

#loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#metrics = [BinaryAccuracy(name='accuracy')]
#model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

# Define a callback to save the best model
checkpoint_callback_train = ModelCheckpoint(
    "best_train_model.h5",  # Filepath to save the best model
    monitor="loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="min",  # Minimize the monitored metric 
    verbose=1  # Display progress
)

# Define a callback to save the best model
checkpoint_callback_val = ModelCheckpoint(
    "best_val_model.h5",  # Filepath to save the best model
    monitor="val_loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="min",  # Minimize the monitored metric 
    verbose=1  # Display progress
)

# Model Summary
model.summary()

## Training the model
#model.fit(
#    train_sequences,
#    train_labels,
#    batch_size=1024,
#    epochs=10000,
#    validation_data=(validation_sequences, validation_labels),
#    shuffle=True,
#    callbacks=[checkpoint_callback_train, checkpoint_callback_val]
#)

model.load_weights("best_val_model.h5")
predictions = model.predict(test_sequences)

predictions = predictions.flatten()


correct = 0
for pred, lab in zip(predictions, test_labels):
    print(f"{pred:.4f} {lab:.4f}")
    if pred < 0 and lab < 0:
        correct += 1
    elif pred > 0 and lab > 0:
        correct += 1

print(f"{correct/len(predictions)=}")



