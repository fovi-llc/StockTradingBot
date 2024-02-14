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

SEQUENCE_LEN = 14

tickers = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","ADBE","CRM","ORCL","INTC","IBM",]
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

corr_sum = []
for ticker in tickers:
    data = yf.download(ticker, period="60d", interval="5m")

    # Calculate the technical indicators
    close = data['Close']
    upper, lower = calculate_bollinger_bands(close, window=SEQUENCE_LEN, num_of_std=2)
    width = upper - lower
    rsi = calculate_rsi(close, window=SEQUENCE_LEN)
    roc = calculate_roc(close, periods=SEQUENCE_LEN)
    volume = data['Volume']
    diff = data['Close'].diff(1)
    percent_change_close = data['Close'].pct_change() * 100

    # Create a DataFrame for the current ticker
    ticker_df = pd.DataFrame({
        ticker+'_close': close,
        ticker+'_rsi': rsi,
        ticker+'_roc': roc,
        ticker+'_volume': volume,
        ticker+'_diff': diff,
        ticker+'_width': width,
        ticker+'_percent_change_close': percent_change_close,
    }).dropna()  # Drop rows with NaN values resulting from diff and pct_change

    # Normalize the features
    MEAN = ticker_df.mean()
    STD = ticker_df.std()
    ticker_df = (ticker_df - MEAN) / STD

    # Set threshold
    Q1 = ticker_df.quantile(0.005)
    Q3 = ticker_df.quantile(0.995)
    IQR = Q3 - Q1

    # Define a mask to filter out outliers
    mask = ~((ticker_df < (Q1 - 1.5 * IQR)) | (ticker_df > (Q3 + 1.5 * IQR))).any(axis=1)

    # Apply the mask to your dataframe to remove outliers
    cleaned_df = ticker_df[mask]

    # Calculate the correlation matrix
    correlation = cleaned_df.corr()

    # Sum the absolute correlations with the ticker's close, excluding its own correlation
    value = np.sum(np.abs(correlation[ticker+'_close'].drop(ticker+'_close')))
    print(ticker, value)
    corr_sum.append(value)

# Find the index of the highest sum of correlations
best_stock_index = np.argmax(corr_sum)
best_stock = tickers[best_stock_index]

print(f"Best stock is {best_stock}, {corr_sum[best_stock_index]}")

tickers = []
tickers.append(best_stock)

    
# List to hold data for each ticker
ticker_data_frames = []
stats = {}
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
    #print("MEAN:\n", MEAN)
    #print("STANDARD DEVIATION:\n",STD)

    for column in MEAN.index:
        mean_key = f"{column}_mean"
        std_key = f"{column}_std"
        stats[mean_key] = MEAN[column]
        stats[std_key] = STD[column]
    
    # Normalize the training features
    ticker_df = (ticker_df - MEAN) / STD

    MIN = ticker_df.min()
    MAX = ticker_df.max()
    #print("MIN:\n", MIN)
    #print("MAX:\n", MAX)
    
    ## Normalize the training features
    #ticker_df = (ticker_df - MIN) / (MAX - MIN)
    
    print(ticker_df.describe())
    print(ticker_df.corr())

    pairplot = sns.pairplot(ticker_df)
    plt.savefig(f"pairplot/{ticker}_pairplot.png")
    plt.close()
    #exit()
    #ticker_data_frames.append(ticker_df)

    # Set threshold
    Q1 = ticker_df.quantile(0.005)
    Q3 = ticker_df.quantile(0.995)
    IQR = Q3 - Q1

    # Define a mask to filter out outliers
    mask = ~((ticker_df < (Q1 - 1.5 * IQR)) | (ticker_df > (Q3 + 1.5 * IQR))).any(axis=1)

    # Apply the mask to your dataframe to remove outliers
    cleaned_df = ticker_df[mask]

    print(cleaned_df.corr())
    
    # Now you can save the cleaned pairplot
    pairplot_cleaned = sns.pairplot(cleaned_df)
    pairplot_cleaned.savefig(f"pairplot/{ticker}_pairplot_cleaned.png")
    plt.close()
    # exit()
    ticker_data_frames.append(cleaned_df)

# Concatenate all ticker DataFrames
percent_change_data = pd.concat(ticker_data_frames, axis=1)
print(stats)

# Remove any NaN values that may have occurred from the pct_change() calculation
percent_change_data.replace([np.inf, -np.inf], np.nan, inplace=True)
percent_change_data.dropna(inplace=True)

print(percent_change_data)

# Function to create X-day sequences for each ticker
def create_sequences(data, datetime, labels, sequence_length=SEQUENCE_LEN):
    sequences = []
    lab = []
    data_size = len(data)
    for i in range(data_size - sequence_length + 1):
        times = datetime[i:i + sequence_length]
        sequence_ok = True  # A flag to indicate if the sequence is valid
        
        # Check if all consecutive timestamps are 5 minutes apart
        for j in range(1, len(times)):
            delta = times[j] - times[j-1]
            if delta.total_seconds() != 300:
                sequence_ok = False  # Invalidate the sequence
                break  # No need to check further timestamps
        
        # Only append the sequence if all timestamps are 5 minutes apart
        if sequence_ok:
            sequences.append(data[i:i + sequence_length])
            lab.append([labels[i-1], labels[i]])
            
    return np.array(sequences), np.array(lab)

# Example trading strategy for labeling
def define_label(data):
    #data = (data * STD[f'{tickers[0]}_close']) + MEAN[f'{tickers[0]}_close']
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
    print(ticker_data)
    datetime = percent_change_data.index.to_numpy()
    print(datetime)
    ticker_sequences, lab = create_sequences(ticker_data, datetime, labels[ticker+'_close'].values[SEQUENCE_LEN-1:])
    print(f"{len(ticker_sequences)=}, {len(lab)=}, {len(ticker_sequences)==len(lab)=}")
    sequences_dict[ticker] = ticker_sequences

    # Align labels with sequences
    sequence_labels[ticker] = lab #labels[ticker+'_close'].values[SEQUENCE_LEN-1:]

# Combine data and labels from all tickers
all_sequences = []
all_labels = []

for ticker in tickers:
    all_sequences.extend(sequences_dict[ticker])
    all_labels.extend(sequence_labels[ticker])

# Convert to numpy arrays
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)
#all_labels = np.array([[label, label] for label in all_labels])

print(all_labels, all_labels.shape)

#count = {}
#for l in all_labels:
#    if l not in count.keys():
#        count[l] = 0
#    count[l] += 1
#for key in count.keys():
#    print(key, count[key])

# Shuffle
np.random.seed(42)
shuffled_indices = np.random.permutation(len(all_sequences))
unshuffle_indices = np.argsort(shuffled_indices)
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

DROPOUT = 0.05
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
#hidden_dense =  Dense(10)(output1)
#output2 = Dense(1)(hidden_dense)
#output = Concatenate()([output1, output2])
#output = Dense(1, activation='softmax')(dense_out)

# Convert them to tensors
MEAN_tensor = tf.constant(MEAN[f'{tickers[0]}_close'], dtype=tf.float32)
STD_tensor = tf.constant(STD[f'{tickers[0]}_close'], dtype=tf.float32)

def custom_loss(y_true, y_pred):
    y_true = (y_true * STD_tensor) + MEAN_tensor
    y_pred = (y_pred * STD_tensor) + MEAN_tensor
    #return tf.reduce_mean(tf.square(y_true - y_pred))
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def direction_sensitive_loss_01(y_true, y_pred):
    # Denormalize predictions and true values
    y_true_denorm = (y_true * STD_tensor) + MEAN_tensor
    y_pred_denorm = (y_pred * STD_tensor) + MEAN_tensor
    
    # Calculate the change in true values and predicted values
    # true_change reflects the actual movement from prev_value to current_value in y_true
    # pred_change reflects the movement from prev_value in y_true to current_value in y_pred
    true_change = y_true_denorm[:, 1] - y_true_denorm[:, 0]
    pred_change = y_pred_denorm[:, 1] - y_true_denorm[:, 0]
    
    # Determine if the prediction got the direction right or wrong
    direction_correct = tf.equal(tf.sign(true_change), tf.sign(pred_change))
    
    # Convert boolean to float for calculation (True -> 1.0, False -> 2.0 to penalize wrong direction more)
    direction_penalty = tf.where(direction_correct, 1.0, 2.0)
    
    # Calculate the absolute error for the current value
    abs_error = tf.abs(y_true_denorm[:, 1] - y_pred_denorm[:, 1])
    
    # Apply direction penalty to the absolute error
    penalized_error = abs_error * direction_penalty
    
    # Return the mean of the penalized error
    return tf.reduce_mean(penalized_error)

def differentiable_direction_sensitive_loss(y_true, y_pred):
    y_true_denorm = (y_true * STD_tensor) + MEAN_tensor
    y_pred_denorm = (y_pred * STD_tensor) + MEAN_tensor
    
    # Calculate changes
    true_change = y_true_denorm[:, 1] - y_true_denorm[:, 0]
    pred_change = y_pred_denorm[:, 0] - y_true_denorm[:, 0]
    
    # Dot product between true change and predicted change
    # This will be positive for correct direction predictions and negative for incorrect ones
    direction_dot_product = true_change * pred_change
    
    # Use a smooth, differentiable function to penalize negative dot products (wrong directions)
    # The exponential function ensures that wrong directions are penalized more heavily than right directions
    #direction_penalty = tf.exp(-direction_dot_product)
    scale_factor = 10.0  # This can be tuned
    direction_penalty = tf.sigmoid(-scale_factor * direction_dot_product)
    #direction_penalty = tf.nn.relu(-scale_factor * direction_dot_product)
    
    # Calculate the absolute error for the predicted next value
    abs_error = tf.abs(y_true_denorm[:, 1] - y_pred_denorm[:, 0])
    
    # Combine absolute error with directional penalty
    #penalized_error = abs_error * direction_penalty
    penalized_error = abs_error + (direction_penalty - 0.5)
    
    return tf.reduce_mean(penalized_error)


def dir_acc(y_true, y_pred):
    y_true_denorm = (y_true * STD_tensor) + MEAN_tensor
    y_pred_denorm = (y_pred * STD_tensor) + MEAN_tensor
    true_change = y_true_denorm[:, 1] - y_true_denorm[:, 0]
    pred_change = y_pred_denorm[:, 0] - y_true_denorm[:, 0]
    correct_direction = tf.equal(tf.sign(true_change), tf.sign(pred_change))
    return tf.reduce_mean(tf.cast(correct_direction, tf.float32))


# Build and compile the model
model = Model(inputs=[inputs], outputs=output)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=differentiable_direction_sensitive_loss, metrics=[dir_acc, 'mae'])

#model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
#model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.summary()

# Define a callback to save the best model
checkpoint_callback_train = ModelCheckpoint(
    "best_train_model.h5",  # Filepath to save the best model
    monitor="dir_acc",  #"loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric 
    verbose=1,  # Display progress
)

# Define a callback to save the best model
checkpoint_callback_val = ModelCheckpoint(
    "best_val_model.h5",  # Filepath to save the best model
    monitor="val_dir_acc", #"val_loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric 
    verbose=1,  # Display progress
)



# Load Weights
#model.load_weights("best_val_model.h5")

# Train Model
model.fit(train_sequences, train_labels,
          validation_data=(validation_sequences, validation_labels),
          epochs=500,  # Adjust based on convergence
          batch_size=16,
          shuffle=True,
          callbacks=[checkpoint_callback_train, checkpoint_callback_val])



model.load_weights("best_val_model.h5")
# Make predictions
test = model.predict(test_sequences)


close_price_mean = MEAN[f'{tickers[0]}_close']
close_price_std = STD[f'{tickers[0]}_close']
close_price_min = MIN[f'{tickers[0]}_close']
close_price_max = MAX[f'{tickers[0]}_close']

good = 0
test = (test * close_price_std) + close_price_mean
test_labels = (test_labels * close_price_std) + close_price_mean

#test = (test * (close_price_max - close_price_min)) + close_price_min
#test_labels = (test_labels * (close_price_max - close_price_min)) + close_price_min

for seq, pred, lab in zip(test_sequences, test, test_labels):
    pred = pred[-1]
    lab = lab[-1]
    close = (seq[-1][0] * close_price_std) + close_price_mean
    #close = (seq[-1][0] * (close_price_max - close_price_min)) + close_price_min
    if pred > close and lab > close:
        good += 1
        print(close, pred, lab, "GOOD")
    elif pred < close and lab < close:
        good += 1
        print(close, pred, lab, "GOOD")
    else: print(close, pred, lab)
        

print(good/len(test))



plt.figure(figsize=(10, 6))
plt.plot(test_labels, label='Actual')
plt.plot(test, label='Predicted')
plt.title('Actual vs. Predicted Testues')
plt.legend()
plt.show()

# Calculate additional metrics as needed
from sklearn.metrics import r2_score

r2 = r2_score(test_labels, test)
print(f"R-squared: {r2}")
