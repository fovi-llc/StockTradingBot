import tqdm
import math
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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Permute, Reshape, Multiply, Flatten

SEQUENCE_LEN = 20

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

#####################
## PICK BEST STOCK ##
#####################
corr_sum = []
for ticker in tickers:
    data = yf.download(ticker, period="60d", interval="5m")

    # Calculate the daily percentage change
    close = data['Close']
    upper, lower = calculate_bollinger_bands(close, window=14, num_of_std=2)
    width = upper - lower
    rsi = calculate_rsi(close, window=14)
    sma = calculate_sma(close, window=14).ewm(span=14).mean()
    roc = calculate_roc(close, periods=14)
    momentum = calculate_momentum(close, periods=14)
    volume = data['Volume']
    diff = data['Close'].diff(1)
    percent_change_close = data['Close'].pct_change() * 100

    # Create a DataFrame for the current ticker and append it to the list
    ticker_df = pd.DataFrame({
        ticker+'_close': close,
        #ticker+'_upper': upper,
        #ticker+'_lower': lower,
        ticker+'_width': width,
        ticker+'_rsi': rsi,
        #ticker+'_sma': sma,
        ticker+'_roc': roc,
        #ticker+'_momentum': momentum,
        ticker+'_volume': volume,
        ticker+'_diff': diff,
        #ticker+'_percent_change_close': percent_change_close,
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
#tickers.append(best_stock)
tickers.append("AAPL")

tickers = ['AAPL', 'ABBV', 'ACN',] #'ADBE','AEP','AFL','AIG','ALGN', ]
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
def create_sequences(data, datetime, labels, mean, std, sequence_length=SEQUENCE_LEN):
    sequences = []
    lab = []
    data_size = len(data)
    #datetime = data['Datetime'].values

    for i in range(data_size - (sequence_length + 12)):
        times = datetime[i:i + sequence_length]
        sequence_ok = True  # A flag to indicate if the sequence is valid
        
        # Calculate time differences as total_seconds
        # Need to ensure times is in a format that allows for subtraction and total_seconds calculation
        #for j in range(1, len(times)):
        #    delta = (times[j] - times[j-1]).astype('timedelta64[s]').astype(int)
        #    if delta != 300:
        #        sequence_ok = False
        #        break
        
        # Only append the sequence if all timestamps are 5 minutes apart
        if sequence_ok:
            sequences.append(data[i:i + sequence_length])
            lab.append([labels[i-1], labels[i+13], mean[0], std[0]])

    #for i in range(10):
    #    i *= -1
    #    print(f'{sequences[i]=}')
    #    print(f'{lab[i]=}')
        
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
                                             datetime,
                                             labels[attribute].values[SEQUENCE_LEN-1:],
                                             stats[attribute+"_mean"].values,
                                             stats[attribute+"_std"].values)
    print(f"{ticker}, {len(ticker_sequences)=}, {len(lab)=}, {len(ticker_sequences)==len(lab)=}")
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


#############
## SHUFFLE ##
#############
np.random.seed(42)
shuffled_indices = np.random.permutation(len(all_sequences))
all_sequences = all_sequences[shuffled_indices]
all_labels = all_labels[shuffled_indices]

train_size = int(len(all_sequences) * 0.9)

# Split sequences
train_sequences = all_sequences[:train_size]
train_labels    = all_labels[:train_size]

other_sequences = all_sequences[train_size:]
other_labels    = all_labels[train_size:]

shuffled_indices = np.random.permutation(len(other_sequences))
other_sequences = other_sequences[shuffled_indices]
other_labels = other_labels[shuffled_indices]

val_size = int(len(other_sequences) * 0.5)

validation_sequences = other_sequences[:val_size]
validation_labels = other_labels[:val_size]

test_sequences = other_sequences[val_size:]
test_labels = other_labels[val_size:]

###########################
## DEFINE NEURAL NETWORK ##
###########################
def attention_layer(inputs, name, reg=1e-4):
    # Here, inputs should have shape [batch, time_steps, input_dim]
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1), name='permute_' + name)(inputs)
    a = Dense(inputs.shape[1], activation='softmax', kernel_regularizer=l2(reg), name='attention_probs_' + name)(a)
    a_probs = Permute((2, 1), name='attention_vec_' + name)(a)
    output_attention_mul = Multiply(name='attention_mul_' + name)([inputs, a_probs])
    return output_attention_mul

DROPOUT = 0.2
reg_strength = 1e-10
# Model architecture
input_shape = train_sequences.shape[1:]
inputs = Input(shape=input_shape)

lstm_out_1 = LSTM(70, return_sequences=True, dropout=DROPOUT, kernel_regularizer=l2(reg_strength))(inputs)
attention_out_1 = attention_layer(lstm_out_1, 'first', reg=reg_strength)
lstm_out_2 = LSTM(60, return_sequences=True, dropout=DROPOUT, kernel_regularizer=l2(reg_strength))(attention_out_1)
attention_out_2 = attention_layer(lstm_out_2, 'second', reg=reg_strength)
lstm_out_3 = LSTM(50, return_sequences=True, dropout=DROPOUT, kernel_regularizer=l2(reg_strength))(attention_out_2)
attention_out_3 = attention_layer(lstm_out_3, 'third', reg=reg_strength)
lstm_out_4 = LSTM(50, return_sequences=True, dropout=DROPOUT, kernel_regularizer=l2(reg_strength))(attention_out_3)
attention_out_4 = attention_layer(lstm_out_4, 'fourth', reg=reg_strength)
lstm_out_5 = LSTM(50, return_sequences=True, dropout=DROPOUT, kernel_regularizer=l2(reg_strength))(attention_out_4)
attention_out_5 = attention_layer(lstm_out_5, 'fith', reg=reg_strength)
lstm_out_6 = LSTM(50, return_sequences=True, dropout=DROPOUT, kernel_regularizer=l2(reg_strength))(attention_out_5)
attention_out_6 = attention_layer(lstm_out_6, 'sixth', reg=reg_strength)

flat_out = Flatten()(attention_out_2)
dense_out = Dense(50, activation='relu', kernel_regularizer=l2(reg_strength))(flat_out)
dense_out = Dropout(DROPOUT)(dense_out)
output = Dense(1,kernel_regularizer=l2(reg_strength))(dense_out)

# Build the model
model = Model(inputs=[inputs], outputs=output)

##########################
## CUSTOM LOSS FUNCTION ##
##########################

def custom_loss(y_true, y_pred):
    y_true = (y_true * STD_tensor) + MEAN_tensor
    y_pred = (y_pred * STD_tensor) + MEAN_tensor
    #return tf.reduce_mean(tf.square(y_true - y_pred))
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def directional_penalty_loss(y_true, y_pred):
    mean, std = tf.cast(y_true[:, 2], tf.float64), tf.cast(y_true[:, 3], tf.float64)
    
    y_true_prev = (tf.cast(y_true[:, 0], tf.float64) * std) + mean
    y_true_next = (tf.cast(y_true[:, 1], tf.float64) * std) + mean
    y_pred_next = (tf.cast(y_pred[:, 0], tf.float64) * std) + mean
    
    true_change = y_true_next - y_true_prev
    pred_change = y_pred_next - y_true_prev
    
    direction_dot_product = true_change * pred_change
    direction_penalty = tf.sigmoid(-1 * direction_dot_product)
    
    return tf.reduce_mean(direction_penalty)

def custom_mae_loss(y_true, y_pred):
    y_true_next = tf.cast(y_true[:, 1], tf.float64)
    y_pred_next = tf.cast(y_pred[:, 0], tf.float64)
    abs_error = tf.abs(y_true_next - y_pred_next)
    
    return tf.reduce_mean(abs_error)

def differentiable_direction_sensitive_loss(y_true, y_pred):
    mean, std = y_true[:, 2], y_true[:, 3]
    
    y_true_prev = (y_true[:, 0] * std) + mean
    y_true_next = (y_true[:, 1] * std) + mean
    y_pred_next = (y_pred[:, 0] * std) + mean
    
    true_change = y_true_next - y_true_prev
    pred_change = y_pred_next - y_true_prev
    
    direction_dot_product = true_change * pred_change
    
    direction_penalty = tf.sigmoid(-1 * direction_dot_product)
    
    abs_error = tf.abs(y_true_next - y_pred_next)
    abs_error = tf.abs(y_true[:, 1] - y_pred[:, 0])
    
    penalized_error = abs_error + direction_penalty
    
    return tf.reduce_mean(penalized_error)

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
EPOCHS = 150
BATCH_SIZE = 16
num_batches = len(train_sequences) // BATCH_SIZE
print(num_batches)

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.01,
    decay_steps=num_batches * EPOCHS,
    end_learning_rate=0.0001,
    power=1.0,  # Linear decay
    cycle=False  # Not a cyclical schedule; it will not restart once it reaches the end
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=custom_mae_loss, metrics=[dir_acc])

model.summary()

# Define a callback to save the best model
checkpoint_callback_train = ModelCheckpoint(
    "best_train_model.keras",  # Filepath to save the best model
    monitor="dir_acc",  #"loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric 
    verbose=1,  # Display progress
)

# Define a callback to save the best model
checkpoint_callback_val = ModelCheckpoint(
    "best_val_model.keras",  # Filepath to save the best model
    monitor="val_dir_acc", #"val_loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric 
    verbose=1,  # Display progress
)

def get_lr_callback(batch_size=16, mode='cos', epochs=500, plot=False):
    lr_start, lr_max, lr_min = 0.0001, 0.005, 0.0001  # Adjust learning rate boundaries
    lr_ramp_ep = int(0.30 * epochs)  # 30% of epochs for warm-up
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
model.load_weights("best_val_model.keras")

# Train Model
model.fit(train_sequences, train_labels,
          validation_data=(validation_sequences, validation_labels),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          shuffle=True,
          callbacks=[checkpoint_callback_train, checkpoint_callback_val, get_lr_callback(batch_size=BATCH_SIZE, epochs=EPOCHS)])

batch_size = 32  # Define your batch size
# Assuming X_train and y_train are your features and labels respectively
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels)).batch(batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices((validation_sequences, validation_labels)).batch(batch_size)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels)).batch(batch_size)


@tf.function(reduce_retracing=True)
def train_step(inputs, labels, loss_function):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    dir_accuracy = dir_acc(labels, predictions)
    return loss, dir_accuracy

@tf.function
def valid_step(inputs, labels, loss_function):
    predictions = model(inputs, training=False)
    loss = loss_function(labels, predictions)
    dir_accuracy = dir_acc(labels, predictions)
    return loss, dir_accuracy

# Custom training loop
best_val_dir_acc = 0.0
best_epoch = 0

#EPOCHS = 10000
#for epoch in range(EPOCHS):
#    print("\nStart of epoch %d" % (epoch,))
#    
#    # Alternate between loss functions
#    if epoch % 2 == 0:
#        current_loss_function = directional_penalty_loss
#    else:
#        current_loss_function = custom_mae_loss
#    
#    # Training loop
#    total_loss = 0
#    total_dir_acc = 0
#    num_batches = 0
#    for x_batch_train, y_batch_train in tqdm.tqdm(train_dataset):
#        loss, dir_accuracy = train_step(x_batch_train, y_batch_train, current_loss_function)
#        total_loss += loss
#        total_dir_acc += dir_accuracy
#        num_batches += 1
#    train_loss = total_loss / num_batches
#    train_dir_acc = total_dir_acc / num_batches
#    
#
#    # Validation loop
#    total_loss_val = 0
#    total_dir_acc_val = 0
#    num_batches_val = 0
#    for x_batch_val, y_batch_val in valid_dataset:
#        loss, dir_accuracy = valid_step(x_batch_val, y_batch_val, current_loss_function)
#        total_loss_val += loss
#        total_dir_acc_val += dir_accuracy
#        num_batches_val += 1
#    valid_loss = total_loss_val / num_batches
#    val_dir_acc = total_dir_acc_val / num_batches_val
#
#    # Check if the current epoch's validation directional accuracy is the best so far
#    if val_dir_acc > best_val_dir_acc:
#        print(f"Validation directional accuracy improved from {best_val_dir_acc:.4f} to {val_dir_acc:.4f}")
#        best_val_dir_acc = val_dir_acc
#        best_epoch = epoch
#        model.save_weights("best_val_model.keras")
#
#    print(f"{train_loss=:.4f}, {train_dir_acc=:.4f}, {valid_loss=:.4f}, {val_dir_acc=:.4f}")
#    print(f"{best_val_dir_acc=:.4f} at epoch {best_epoch}")



########################
## INFER MODEL (TEST) ##
########################
# Load Weights
model.load_weights("best_train_model.keras")

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
