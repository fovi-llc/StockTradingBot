import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Permute, Reshape, Multiply, Flatten
from tensorflow.keras.models import Model

SEQUENCE_LEN = 10

tickers = [
    "AAPL",]#"MSFT","GOOGL","AMZN","TSLA","META","NVDA","ADBE","CRM","ORCL","INTC","IBM",
#    "QCOM","CSCO","ASML","TXN","AMD","SAP","SHOP","AVGO","INTU","SNOW","SQ","ZM","NFLX",  
#    "PYPL","GOOG","MS","V","MA","JPM","GS","WMT","TGT","HD","LOW","NKE","DIS",
#    "CMCSA","PEP","KO","T","VZ","AAP","F",
#]

def calculate_momentum(data, periods=10):
    """Calculate Momentum."""
    momentum = data - data.shift(periods)
    return momentum

def calculate_roc(data, periods=10):
    """Calculate Rate of Change."""
    roc = ((data - data.shift(periods)) / data.shift(periods)) * 100
    return roc

def calculate_sma(data, window=10):
    """Calculate Simple Moving Average."""
    sma = data.rolling(window=window).mean()
    return sma

def calculate_rsi(data, window=10):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=10, num_of_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band


# List to hold data for each ticker
ticker_data_frames = []

for ticker in tickers:
    # Download historical data for the ticker
    #data = yf.download(ticker, start=start_date, end=end_date)
    #data = yf.download(ticker, period="10y", interval="1d")
    #data = yf.download(ticker, period="60d", interval="90m")
    #data = yf.download(ticker, period="730d", interval="60m")
    #data = yf.download(ticker, period="60d", interval="30m")
    #data = yf.download(ticker, period="60d", interval="15m")
    data = yf.download(ticker, period="60d", interval="5m")
    #data = yf.download(ticker, period="7d", interval="1m")
    print(data)

    scaler = StandardScaler()

    # Calculate the daily percentage change
    close = data['Close']
    upper, lower = calculate_bollinger_bands(close, window=SEQUENCE_LEN, num_of_std=2)
    width = upper - lower
    rsi = calculate_rsi(close, window=SEQUENCE_LEN)
    sma = calculate_sma(close, window=SEQUENCE_LEN).ewm(span=SEQUENCE_LEN).mean()
    roc = calculate_roc(close, periods=SEQUENCE_LEN)
    momentum = calculate_momentum(close, periods=SEQUENCE_LEN)
    volume = data['Volume']

    diff = data['Close'].diff(1)
    
    percent_change_close = data['Close'].pct_change() * 100
    percent_upper, percent_lower = calculate_bollinger_bands(percent_change_close)
    percent_rsi = calculate_rsi(percent_change_close)
    percent_sma = calculate_sma(percent_change_close)
    percent_roc = calculate_roc(percent_change_close)
    percent_momentum = calculate_momentum(percent_change_close)

    percent_change_volume = data["Volume"].pct_change() * 100

    # Create a DataFrame for the current ticker and append it to the list
    ticker_df = pd.DataFrame({
        ticker+'_close': close,
        ticker+'_rsi': rsi,
        ticker+'_roc': roc,
        ticker+'_volume': volume,
        ticker+'_diff': diff,
        ticker+'_width': width,
        ticker+'_percent_change_close': percent_change_close,
    })
    print(ticker_df)
#    ticker_df = pd.DataFrame({
#        ticker+'_close': close,
#        ticker+'_upper': upper,
#        ticker+'_lower': lower,
#        ticker+'_rsi': rsi,
#        ticker+'_sma': sma,
#        ticker+'_roc': roc,
#        ticker+'_momentum': momentum,
#        ticker+'_volume': volume,})
#        ticker+'_percent_close': percent_change_close,
#        ticker+'_percent_upper': percent_upper,
#        ticker+'_percent_lower': percent_lower,
#        ticker+'_percent_rsi': percent_rsi,
#        ticker+'_percent_sma': percent_sma,
#        ticker+'_percent_roc': percent_roc,
#        ticker+'_percent_momentum': percent_momentum,
#        ticker+'_percent_volume' : percent_change_volume
#    })
    
    MEAN = ticker_df.mean()
    STD = ticker_df.std()
    print("MEAN:\n", MEAN)
    print("STANDARD DEVIATION:\n",STD)
    
    
    # Normalize the training features
    ticker_df = (ticker_df - MEAN) / STD

    MIN = ticker_df.min()
    MAX = ticker_df.max()
    print("MIN:\n", MIN)
    print("MAX:\n", MAX)
    #
    ## Normalize the training features
    #ticker_df = (ticker_df - MIN) / (MAX - MIN)
    #
    #print(ticker_df.describe())
    #print(ticker_df.corr())

    pairplot = sns.pairplot(ticker_df)
    plt.savefig(f"{ticker}_pairplot.png")
    plt.close()
    #exit()
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
def define_label(data):
    #data = (data * STD['AAPL_percent_change_close']) + MEAN['AAPL_percent_change_close']
    return data
    
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
#    upper = percent_change_data[ticker+'_upper'].values
#    lower = percent_change_data[ticker+'_lower'].values
    rsi = percent_change_data[ticker+'_rsi'].values
#    sma = percent_change_data[ticker+'_sma'].values
    roc = percent_change_data[ticker+'_roc'].values
#    momentum = percent_change_data[ticker+'_momentum'].values
    volume = percent_change_data[ticker+'_volume'].values
    diff = percent_change_data[ticker+'_diff'].values
    width = percent_change_data[ticker+'_width'].values

#    percent_close = percent_change_data[ticker+'_percent_close'].values
#    percent_upper = percent_change_data[ticker+'_percent_upper'].values
#    percent_lower = percent_change_data[ticker+'_percent_lower'].values
#    percent_rsi = percent_change_data[ticker+'_percent_rsi'].values
#    percent_sma = percent_change_data[ticker+'_percent_sma'].values
#    percent_roc = percent_change_data[ticker+'_percent_roc'].values
#    percent_momentum = percent_change_data[ticker+'_momentum'].values
    
    # Combine close and volume data
    ticker_data = np.column_stack((close,
                                   rsi,
                                   roc,
                                   volume,
                                   diff,
                                   width))
#    ticker_data = np.column_stack((close,
#                                   upper,
#                                   lower,
#                                   rsi,
#                                   sma,))
#                                   #roc,
#                                   #momentum,
#                                   #volume,))
#                                   #percent_close,
#                                   #percent_upper,
#                                   #percent_lower,
#                                   #percent_rsi,
#                                   #percent_sma,
#                                   #percent_roc,
#                                   #percent_momentum))
    
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

count = {}
for l in all_labels:
    if l not in count.keys():
        count[l] = 0
    count[l] += 1
for key in count.keys():
    print(key, count[key])

# Shuffle
np.random.seed(42)
shuffled_indices = np.random.permutation(len(all_sequences))
#all_sequences = all_sequences[shuffled_indices]
#all_labels = all_labels[shuffled_indices]

# Assuming all_sequences is your dataset and all_labels are the corresponding labels
total_samples = len(all_sequences)

train_size = int(total_samples * 0.9)
validation_size = int(total_samples * 0.05)
test_size = total_samples - train_size - validation_size  # Ensures all samples are included

print(f"Train Size: {train_size}, Validation Size: {validation_size}, Test Size: {test_size}")

# Split sequences
train_sequences = all_sequences[:train_size]
validation_sequences = all_sequences[train_size:train_size+validation_size]
test_sequences = all_sequences[train_size+validation_size:]

# Split labels
train_labels = all_labels[:train_size]
validation_labels = all_labels[train_size:train_size+validation_size]
test_labels = all_labels[train_size+validation_size:]

def attention_layer(inputs, name):
    # Here, inputs should have shape [batch, time_steps, input_dim]
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1), name='permute_' + name)(inputs)
    a = Dense(inputs.shape[1], activation='softmax', name='attention_probs_' + name)(a)
    a_probs = Permute((2, 1), name='attention_vec_' + name)(a)
    output_attention_mul = Multiply(name='attention_mul_' + name)([inputs, a_probs])
    return output_attention_mul

DROPOUT = 0.020
# Model architecture
input_shape = train_sequences.shape[1:]  # Adjust based on your dataset
inputs = Input(shape=input_shape)

# First LSTM layer
lstm_out_1 = LSTM(70, return_sequences=True, dropout=DROPOUT)(inputs)

# First attention mechanism
attention_out_1 = attention_layer(lstm_out_1, 'first')

# Second LSTM layer - now with return_sequences=True to maintain the sequence for attention
lstm_out_2 = LSTM(50, return_sequences=True, dropout=DROPOUT)(attention_out_1)

# Second attention mechanism
attention_out_2 = attention_layer(lstm_out_2, 'second')

# Flattening the output of the second attention mechanism to connect to dense layers
flat_out = Flatten()(attention_out_2)

# Dense layer
dense_out = Dense(50, activation='relu')(flat_out)
dense_out = Dropout(DROPOUT)(dense_out)

# Output layer
output = Dense(1,)(dense_out)
#output = Dense(1, activation='softmax')(dense_out)

# Build and compile the model
model = Model(inputs=[inputs], outputs=output)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

#model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.summary()

# Define a callback to save the best model
checkpoint_callback_train = ModelCheckpoint(
    "best_train_model.h5",  # Filepath to save the best model
    monitor="loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="min",  # Minimize the monitored metric 
    verbose=1,  # Display progress
)

# Define a callback to save the best model
checkpoint_callback_val = ModelCheckpoint(
    "best_val_model.h5",  # Filepath to save the best model
    monitor="val_loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="min",  # Minimize the monitored metric 
    verbose=1,  # Display progress
)

## Define a callback to save the best model
#checkpoint_callback_train = ModelCheckpoint(
#    "best_train_model.h5",  # Filepath to save the best model
#    monitor="accruacy",  # Metric to monitor
#    save_best_only=True,  # Save only the best model
#    mode="max",  # Minimize the monitored metric 
#    verbose=1,  # Display progress
#)
#
## Define a callback to save the best model
#checkpoint_callback_val = ModelCheckpoint(
#    "best_val_model.h5",  # Filepath to save the best model
#    monitor="val_accuracy",  # Metric to monitor
#    save_best_only=True,  # Save only the best model
#    mode="max",  # Minimize the monitored metric 
#    verbose=1,  # Display progress
#)



correct_percentage = []
win_percentage = []
while len(validation_sequences) != 0:
    

    model = Model(inputs=[inputs], outputs=output)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    #model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Define a callback to save the best model
    checkpoint_callback_val = ModelCheckpoint(
        "best_val_model.h5",  # Filepath to save the best model
        monitor="val_loss",  # Metric to monitor
        save_best_only=True,  # Save only the best model
        mode="min",  # Minimize the monitored metric 
        verbose=1,  # Display progress
    )

    val_seq = train_sequences[-5:]
    val_lab = train_labels[-5:]
    
    model.fit(train_sequences, train_labels,
              validation_data=(val_seq, val_lab),
              epochs=25,  # Adjust based on convergence
              batch_size=16,
              shuffle=True,
              callbacks=[checkpoint_callback_val],
              verbose=1)


    test_seq, test_lab = validation_sequences[0], validation_labels[0]
    validation_sequences = validation_sequences[1:]
    validation_labels = validation_labels[1:]

    test_seq = np.expand_dims(test_seq, axis=0)
    test_lab = np.expand_dims(test_lab, axis=0)
    train_sequences = np.append(train_sequences, test_seq, axis=0)
    train_labels = np.append(train_labels, test_lab, axis=0)

    model.load_weights("best_val_model.h5")
    
    #test_seq, test_lab = validation_sequences[0], validation_labels[0]
    #test_seq = np.expand_dims(test_seq, axis=0)
    #test_lab = np.expand_dims(test_lab, axis=0)

    predictions = model.predict(test_seq)

#    good = 0
#    for pred, lab in zip(predictions, test_lab):
#        
#        if  np.argmax(pred) == 0 and lab == 0:
#            good += 1
#            print(f"{pred=}, {np.argmax(pred)=}, {lab=}, GOOD!")
#        elif np.argmax(pred) > 0 and lab > 0:
#            good += 1
#            print(f"{pred=}, {np.argmax(pred)=}, {lab=}, GOOD!")
#            
#        else: print(f"{pred=}, {np.argmax(pred)=}, {lab=}, BAD!")
#
#        correct_percentage.append(good/len(predictions))
#        print(f"{correct_percentage[-1]=}, {np.average(correct_percentage)=}")

    predictions = (predictions * STD['AAPL_close']) + MEAN['AAPL_close']
    test_lab = (test_lab * STD['AAPL_close']) + MEAN['AAPL_close']

    good = 0
    win = 0
    for seq, pred, lab in zip(test_seq, predictions, test_lab):
        close = (seq[-1][0] * STD['AAPL_close']) + MEAN['AAPL_close']
        if pred > close and lab > close:
            good += 1
            print(close, pred, lab, "GOOD")
        elif pred < close and lab < close:
            good += 1
            print(close, pred, lab, "GOOD")
        else: print(close, pred, lab)

        if pred > close and lab > close:
            win += 1 ; print("WIN")
        elif pred < close:
            win += 1 ; print("WIN")
        
        
        correct_percentage.append(good/len(predictions))
        print(f"{correct_percentage[-1]=}, {np.average(correct_percentage)=}")

        win_percentage.append(win/len(predictions))
        print(f"{win_percentage[-1]=}, {np.average(win_percentage)=}")
print(f"{np.average(correct_percentage)=}")
print(f"{np.average(win_percentage)=}")

exit()

#model.load_weights("best_val_model.h5")
model.fit(train_sequences, train_labels,
          validation_data=(validation_sequences, validation_labels),
          epochs=500,  # Adjust based on convergence
          batch_size=16,
          shuffle=True,
          callbacks=[checkpoint_callback_train, checkpoint_callback_val])



model.load_weights("best_val_model.h5")
# Make predictions
predictions = model.predict(validation_sequences)

pos = 0
neg = 0
for pred, lab in zip(predictions, validation_labels):
    if pred > 0 and lab > 0:
        pos += 1
    if pred < 0 and lab < 0:
        neg += 1

print((pos + neg) / len(predictions))


close_price_mean = MEAN['AAPL_close']
close_price_std = STD['AAPL_close']
close_price_min = MIN['AAPL_close']
close_price_max = MAX['AAPL_close']

good = 0
predictions = (predictions * close_price_std) + close_price_mean
validation_labels = (validation_labels * close_price_std) + close_price_mean

#predictions = (predictions * (close_price_max - close_price_min)) + close_price_min
#validation_labels = (validation_labels * (close_price_max - close_price_min)) + close_price_min

for seq, pred, lab in zip(validation_sequences, predictions, validation_labels):
    close = (seq[-1][0] * close_price_std) + close_price_mean
    #close = (seq[-1][0] * (close_price_max - close_price_min)) + close_price_min
    if pred > close and lab > close:
        good += 1
        print(close, pred, lab, "GOOD")
    elif pred < close and lab < close:
        good += 1
        print(close, pred, lab, "GOOD")
    else: print(close, pred, lab)
        

print(good/len(predictions))



plt.figure(figsize=(10, 6))
plt.plot(validation_labels, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs. Predicted Validationues')
plt.legend()
plt.show()

# Calculate additional metrics as needed
from sklearn.metrics import r2_score

r2 = r2_score(validation_labels, predictions)
print(f"R-squared: {r2}")
