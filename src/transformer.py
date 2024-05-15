import tqdm
import math
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Permute, Reshape, Multiply, Flatten, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D



parser = argparse.ArgumentParser(description='Process some stock tickers.')

parser.add_argument('--tickers',
                    nargs='+',
                    help='List of ticker symbols to process',
                    required=True)

# Parse the arguments
args = parser.parse_args()

tickers = args.tickers
print(tickers)

SEQUENCE_LEN = 24 #20

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

#tickers = ['AAPL','ABBV','ACN', 'ADBE','AEP','AFL','AIG','ALGN', ]
#           'ALL','AMAT','AMD','AMGN','AMZN','AON','APA','APD','APH',
#           'ASML','AVB','AVGO','AXP','AZO','BA','BAC','BBY','BDX','BEN',
#           'BIIB','BK','BKNG','BMY','BSX','BXP','C','CAG','CAH','CAT',
#           'CB','CCL','CDNS','CE','CF','CHD','CHTR','CI','CINF','CL',
#           'CLX','CMA','CMCSA','CME','CMG','CMI','COF','COO','COP',
#           'COST','CPB','CPRT','CRM','CSCO','CSX','CTAS','CTSH', 
#           'CVS','CVX','D','DAL','DD','DE','DFS','DG','DGX','DHI','DHR','DIS',
#           'DLR','DLTR','DOV','DOW','DRI','DTE',
#           'DUK','DVA','DVN','DXC','EA','EBAY','ECL','ED','EFX','EIX','EL',
#           'EMN','EMR','EOG','EQIX','EQR','ES','ESS','ETN','ETR','EW',
#           'EXC','EXPD','EXPE','EXR','F','FANG','FAST','FCX','FDX','FE',
#           'FFIV','FIS','FITB','FLS','FLT','FMC','FOX','FOXA',
#           'FRT','FTI','FTNT','FTV','GD','GE','GILD','GIS','GL','GLW',
#           'GM','GOOG','GOOGL','GPC','GPN','GPS','GRMN','GS','GWW','HAL','HAS',
#           'HBAN','HBI','HCA','HCP','HD','HES','HIG','HII','HLT','HOG','HOLX',
#           'HON','HP','HPE','HPQ','HRB','HRL','HST','HSY','HUM','IBM','ICE',
#           'IDXX','IFF','ILMN','INCY','INTC','INTU','IP','IPG','IPGP',
#           'IQV','IR','IRM','ISRG','IT','ITW','IVZ','JBHT','JCI','JEF',
#           'JKHY','JNJ','JNPR','JPM','JWN','K','KEY','KEYS','KHC','KIM','KLAC',
#           'KMB','KMI','KMX','KO','KR','KSS','L','LEG','LEN','LH',
#           'LHX','LIN','LKQ','LLY','LMT','LNC','LNT','LOW','LRCX','LUV','LW',
#           'LYB','M','MA','MAA','MAC','MAR','MAS','MCD','MCHP','MCK','MCO',
#           'MDLZ','MDT','MET','META','MGM','MHK','MKC','MKTX','MLM','MMC',
#           'MMM','MNST','MO','MOS','MPC','MRK','MRO','MS','MSCI','MSFT','MSI',
#           'MTB','MTD','MU','NCLH','NDAQ','NEE','NEM',
#           'NFLX','NI','NKE','NKTR','NOC','NOV','NRG','NSC','NTAP',
#           'NTRS','NUE','NVDA','NWL','NWS','NWSA','O','OI','OKE','OMC','ORCL',
#           'ORLY','OXY','PAYX','PCAR','PEG','PEP','PFE','PFG','PG',
#           'PGR','PH','PHM','PKG','PLD','PM','PNC','PNR','PNW','PPG',
#           'PPL','PRGO','PRU','PSA','PSX','PVH','PWR','PXD','PYPL','QCOM',
#           'QRVO','RCL','REG','REGN','RF','RHI','RJF','RL','RMD','ROK',
#           'ROL','ROP','ROST','RSG','RTX','SAP','SBAC','SBUX','SCHW',
#           'SEE','SHOP','SHW','SJM','SLB','SLG','SNA','SNOW','SNPS',
#           'SO','SPG','SPGI','SQ','SRE','STI','STT','STX','STZ','SWK','SWKS',
#           'SYF','SYK','SYY','T','TAP','TDG','TEL','TFX','TGT','TJX',
#           'TMO','TMUS','TPR','TRIP','TROW','TRV','TSCO','TSLA','TSN',
#           'TTWO','TXN','TXT','UA','UAA','UAL','UDR','UHS','ULTA','UNH',
#           'UNM','UNP','UPS','URI','USB','V','VFC','VLO',
#           'VMC','VNO','VRSK','VRSN','VRTX','VTR','VZ','WAB','WAT','WBA','WDC',
#           'WEC','WELL','WFC','WHR','WM','WMB','WMT','WRK','WU','WY',
#           'WYNN','XEL','XOM','XRAY','XRX','XYL','YUM','ZBH',
#           'ZION','ZM','ZTS']
    

###################
## DOWNLOAD DATA ##
###################
# List to hold data for each ticker
#stats = {}
#for ticker in tickers:
#    # Download historical data for the ticker
#    #data = yf.download(ticker, start=start_date, end=end_date)
#    #data = yf.download(ticker, period="10y", interval="1d")
#    #data = yf.download(ticker, period="60d", interval="90m")
#    #data = yf.download(ticker, period="730d", interval="60m")
#    #data = yf.download(ticker, period="60d", interval="30m")
#    #data = yf.download(ticker, period="60d", interval="15m")
#    data = yf.download(ticker, period="60d", interval="5m")
#    #data = yf.download(ticker, period="7d", interval="1m")
#
#    # Calculate the daily percentage change
#    close = data['Close']
#    upper, lower = calculate_bollinger_bands(close, window=14, num_of_std=2)
#    width = upper - lower
#    rsi = calculate_rsi(close, window=14)
#    sma = calculate_sma(close, window=14).ewm(span=14).mean()
#    roc = calculate_roc(close, periods=14)
#    momentum = calculate_momentum(close, periods=14)
#    volume = data['Volume']
#    diff = data['Close'].diff(1)
#    percent_change_close = data['Close'].pct_change() * 100
#
#    # Create a DataFrame for the current ticker and append it to the list
#    ticker_df = pd.DataFrame({
#        ticker+'_close': close,
#        ticker+'_upper': upper,
#        ticker+'_lower': lower,
#        ticker+'_width': width,
#        ticker+'_rsi': rsi,
#        ticker+'_sma': sma,
#        ticker+'_roc': roc,
#        ticker+'_momentum': momentum,
#        ticker+'_volume': volume,
#        ticker+'_diff': diff,
#        ticker+'_percent_change_close': percent_change_close,
#    })
#    
#    MEAN = ticker_df.mean()
#    STD = ticker_df.std()
#    MIN = ticker_df.min()
#    MAX = ticker_df.max()
#
#    for column in MEAN.index:
#        mean_key = f"{column}_mean"
#        std_key = f"{column}_std"
#        stats[mean_key] = MEAN[column]
#        stats[std_key] = STD[column]
#    
#    # Normalize the training features
#    ticker_df = (ticker_df - MEAN) / STD
#
#    #pairplot = sns.pairplot(ticker_df)
#    #plt.savefig(f"pairplot/{ticker}_pairplot.png")
#    #plt.close()
#    #ticker_data_frames.append(ticker_df)
#
#    # Set threshold
#    Q1 = ticker_df.quantile(0.005)
#    Q3 = ticker_df.quantile(0.995)
#    IQR = Q3 - Q1
#
#    # Define a mask to filter out outliers
#    mask = ~((ticker_df < (Q1 - 1.5 * IQR)) | (ticker_df > (Q3 + 1.5 * IQR))).any(axis=1)
#
#    # Apply the mask to your dataframe to remove outliers
#    cleaned_df = ticker_df[mask]
#    
#    #pairplot_cleaned = sns.pairplot(cleaned_df)
#    #pairplot_cleaned.savefig(f"pairplot/{ticker}_pairplot_cleaned.png")
#    #plt.close()
#    cleaned_df.to_csv(f"data/{ticker}.csv")
#stats = pd.DataFrame([stats], index=[0])
#stats.to_csv(f"data/STATS.csv", index=False)
#print(stats)
#exit()

stats = pd.read_csv(f"data/STATS.csv")
ticker_data_frames = []
for ticker in tqdm.tqdm(tickers):
    df = pd.read_csv(f"data/{ticker}.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df['Datetime'] = df['Datetime'].dt.tz_localize(None)
    ticker_data_frames.append(df)
    
# Concatenate all ticker DataFrames
percent_change_data = pd.concat(ticker_data_frames, axis=1)
print(stats)

# Remove any NaN values that may have occurred from the pct_change() calculation
percent_change_data.replace([np.inf, -np.inf], np.nan, inplace=True)
percent_change_data.dropna(inplace=True)

print(percent_change_data)

###############################
## CREATE LABELS (TRANSFORM) ##
###############################
# Example trading strategy for labeling
def define_label(data):
    #data = (data * STD[f'{tickers[0]}_close']) + MEAN[f'{tickers[0]}_close']
    return data
    
# Shift the percentage change data to create labels
labels = percent_change_data.shift(-1).map(define_label)

# Drop the last row in both percent_change_data and labels as it won't have a corresponding label
percent_change_data = percent_change_data.iloc[:-1]
labels = labels.iloc[:-1]

################################
## HELPER TO CREATE SEQUENCES ##
################################
# Function to create X-day sequences for each ticker
def create_sequences(data, labels, mean, std, sequence_length=SEQUENCE_LEN):
    sequences = []
    lab = []
    data_size = len(data)

    # Loop to create each sequence and its corresponding label
    for i in range(data_size - (sequence_length + 13)): # Ensure we have data for the label
        if i == 0:
          continue
        sequences.append(data[i:i + sequence_length])  # The sequence of data
        lab.append([labels[i-1], labels[i + 12], mean[0], std[0]]) # The label and scaling factors

#    for i in range(10):
#        #i *= -1
#        print(f'{sequences[i]=}')
#        print(f'{lab[i]=}')

    return np.array(sequences), np.array(lab)


######################
## CREATE SEQUENCES ##
######################
sequences_dict = {}
sequence_labels = {}
for ticker in tickers:

    # Extract close and volume data for the ticker
    close = percent_change_data[ticker+'_close'].values
    upper = percent_change_data[ticker+'_upper'].values
    lower = percent_change_data[ticker+'_lower'].values
    width = percent_change_data[ticker+'_width'].values
    rsi = percent_change_data[ticker+'_rsi'].values
    sma = percent_change_data[ticker+'_sma'].values
    roc = percent_change_data[ticker+'_roc'].values
    momentum = percent_change_data[ticker+'_momentum'].values
    volume = percent_change_data[ticker+'_volume'].values
    diff = percent_change_data[ticker+'_diff'].values
    pct_change = percent_change_data[ticker+'_percent_change_close'].values
    
    # Combine close and volume data
    ticker_data = np.column_stack((close,
                                   #upper,
                                   #lower,
                                   width,
                                   rsi,
                                   #sma,
                                   roc,
                                   #momentum,
                                   volume,
                                   diff,
                                   pct_change))
    
    # Generate sequences
    #datetime = percent_change_data['Datetime'].to_numpy()
    datetime = percent_change_data['Datetime'].values.astype('datetime64[s]')
    attribute=ticker+"_close"
    ticker_sequences, lab = create_sequences(ticker_data,
                                             labels[attribute].values[SEQUENCE_LEN-1:],
                                             stats[attribute+"_mean"].values,
                                             stats[attribute+"_std"].values)
    #print(f"{ticker}, {len(ticker_sequences)=}, {len(lab)=}, {len(ticker_sequences)==len(lab)=}")
    sequences_dict[ticker] = ticker_sequences

    # Align labels with sequences
    sequence_labels[ticker] = lab #labels[ticker+'_close'].values[SEQUENCE_LEN-1:]

## Combine data and labels from all tickers
#all_sequences = []
#all_labels = []
#
#for ticker in tickers:
#    all_sequences.extend(sequences_dict[ticker])
#    all_labels.extend(sequence_labels[ticker])
#
# Convert to numpy arrays
#all_sequences = np.array(all_sequences)
#all_labels = np.array(all_labels)
#
#
##############
### SHUFFLE ##
##############
#np.random.seed(42)
#shuffled_indices = np.random.permutation(len(all_sequences))
#all_sequences = all_sequences[shuffled_indices]
#all_labels = all_labels[shuffled_indices]
#
#train_size = int(len(all_sequences) * 0.9)
#
## Split sequences
#train_sequences = all_sequences[:train_size]
#train_labels    = all_labels[:train_size]
#
#other_sequences = all_sequences[train_size:]
#other_labels    = all_labels[train_size:]
#
#shuffled_indices = np.random.permutation(len(other_sequences))
#other_sequences = other_sequences[shuffled_indices]
#other_labels = other_labels[shuffled_indices]
#
#val_size = int(len(other_sequences) * 0.5)
#
#validation_sequences = other_sequences[:val_size]
#validation_labels = other_labels[:val_size]
#
#test_sequences = other_sequences[val_size:]
#test_labels = other_labels[val_size:]

train_sequences = []
train_labels = []
validation_sequences = []
validation_labels = []
test_sequences = []
test_labels = []

for ticker in tickers:
    sequences = sequences_dict[ticker]
    labels = sequence_labels[ticker]

    total_size = len(sequences)
    train_size = int(total_size * 0.9)
    val_size = int(total_size * 0.05)
    
    train_sequences.extend(sequences[:train_size])
    train_labels.extend(labels[:train_size])
    
    validation_sequences.extend(sequences[train_size:train_size + val_size])
    validation_labels.extend(labels[train_size:train_size + val_size])
    
    test_sequences.extend(sequences[train_size + val_size:])
    test_labels.extend(labels[train_size + val_size:])

# Convert lists to numpy arrays
train_sequences = np.array(train_sequences)
train_labels = np.array(train_labels)
validation_sequences = np.array(validation_sequences)
validation_labels = np.array(validation_labels)
test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)

# Shuffle train sequences and labels
np.random.seed(42)
shuffled_indices = np.random.permutation(len(train_sequences))
train_sequences = train_sequences[shuffled_indices]
train_labels = train_labels[shuffled_indices]


###########################
## DEFINE NEURAL NETWORK ##
###########################
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Add()([x, inputs])

    # Feed Forward Part
    y = LayerNormalization(epsilon=1e-6)(x)
    y = Dense(ff_dim, activation="relu")(y)
    y = Dropout(dropout)(y)
    y = Dense(inputs.shape[-1])(y)
    return Add()([y, x])

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Create multiple layers of the Transformer block
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Final part of the model
    x = GlobalAveragePooling1D()(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    outputs = Dense(1, activation="linear")(x)

    # Compile model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Model parameters
input_shape = train_sequences.shape[1:]
head_size = 256 #128 #32
num_heads = 16 #8 #2
ff_dim = 1024 #512 #64
num_layers = 12 #6 #2
dropout = 0.20

# Build the model
model = build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, dropout)

##########################
## CUSTOM LOSS FUNCTION ##
##########################

def custom_mae_loss(y_true, y_pred):
    y_true_next = tf.cast(y_true[:, 1], tf.float64)
    y_pred_next = tf.cast(y_pred[:, 0], tf.float64)
    abs_error = tf.abs(y_true_next - y_pred_next)
    
    return tf.reduce_mean(abs_error)

def dir_acc(y_true, y_pred):
    mean, std = tf.cast(y_true[:, 2], tf.float64), tf.cast(y_true[:, 3], tf.float64)
    
    y_true_prev = (tf.cast(y_true[:, 0], tf.float64) * std) + mean
    y_true_next = (tf.cast(y_true[:, 1], tf.float64) * std) + mean
    y_pred_next = (tf.cast(y_pred[:, 0], tf.float64) * std) + mean
    
    true_change = y_true_next - y_true_prev
    pred_change = y_pred_next - y_true_prev
    
    correct_direction = tf.equal(tf.sign(true_change), tf.sign(pred_change))
    
    return tf.reduce_mean(tf.cast(correct_direction, tf.float64))


###################
## COMPILE MODEL ##
###################
BATCH_SIZE = 256
SCALE = BATCH_SIZE // 16
EPOCHS = math.ceil( (270000 // SCALE) / math.ceil(len(train_sequences) / BATCH_SIZE) )

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=custom_mae_loss, metrics=[dir_acc])

model.summary()


#################
## CHECKPOINTS ##
#################
# Define a callback to save the best model
checkpoint_callback_train = ModelCheckpoint(
    "transformer_train_model.keras",  # Filepath to save the best model
    monitor="dir_acc",  #"loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric 
    verbose=1,  # Display progress
)

# Define a callback to save the best model
checkpoint_callback_val = ModelCheckpoint(
    "transformer_val_model.keras",  # Filepath to save the best model
    monitor="val_dir_acc", #"val_loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric 
    verbose=1,  # Display progress
)

def get_lr_callback(batch_size=16, mode='cos', epochs=500, plot=False):
    lr_start, lr_max, lr_min = 5e-6, 6e-6 * batch_size, 1e-6  # Adjust learning rate boundaries
    lr_ramp_ep = int(0.25 * epochs)  # 35% of epochs for warm-up
    lr_sus_ep = max(0, int(0.10 * epochs) - lr_ramp_ep)  # Optional sustain phase, adjust as needed

    def lrfn(epoch):
        if epoch < lr_ramp_ep:  # Warm-up phase
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:  # Sustain phase at max learning rate
            lr = lr_max
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        else:
            lr = lr_min  # Default to minimum learning rate if mode is not recognized

        return lr

    if plot:  # Plot learning rate curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Scheduler')
        plt.show()

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    

#################
## TRAIN MODEL ##
#################
try:
    model.load_weights("transformer_val_model.keras")
except Exception as e:
    print(e)

# Train Model
model.fit(train_sequences, train_labels,
          validation_data=(validation_sequences, validation_labels),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          shuffle=True,
          callbacks=[checkpoint_callback_train, checkpoint_callback_val, get_lr_callback(batch_size=BATCH_SIZE, epochs=EPOCHS)])


########################
## INFER MODEL (TEST) ##
########################
# Load Weights
model.load_weights("transformer_val_model.keras")

# Make predictions
accuracy = model.evaluate(train_sequences, train_labels)[1]
print(accuracy)
predictions = model.predict(train_sequences)

plt.figure(figsize=(10, 6))
plt.plot(train_labels[-500:, :2], label='Actual')
plt.plot(predictions[-500:], label='Predicted')
plt.title(f'Actual vs. Predicted Trainues with {accuracy=:.4f}')
plt.legend()
plt.savefig(f"{ticker}_Actual_VS_Predicted.png")
#plt.show()

# Calculate additional metrics as needed
from sklearn.metrics import r2_score

r2 = r2_score(train_labels[:, 1], predictions[:, 0])
print(f"R-squared: {r2}")
