import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import yfinance as yf

from utils import utils, NN, upward_trend

#######################
## find upward trend ##
#######################
#stocks = upward_trend.find_upward()
#ticker = next(iter(stocks))

################
## Parameters ##
################
sequence_len = 10
input_shape = (sequence_len, 4)
batch_size = 32
lr = 0.001

peroid = '75m' #1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
interval = '1m' # 1m, 2m, 5m, 15m, 30m, 60m 90m, 1h, 1d, 5d, 1wk, 1mo
ticker = "NVDA"


####################
## Neural Network ##
####################

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

model = NN.get_model(input_shape)
model.summary()

################
## simulation ##
################
start_price = yf.Ticker(ticker).history().tail(1)['Close'].values[0]
OWN = False
BUY_PRICE = 0.0
TOTAL = 0.0
while True:

    ##########
    ## Data ##
    ##########
    data = utils.get_stock_data(ticker, peroid, interval)
    time_series, labels = utils.make_time_series(data, sequence_len)
   
    train_data = time_series[:, ::]
    train_labels = labels[:, ::]
    
    val_data = time_series[-10:, ::]
    val_labels = labels[-10:, ::]

    prev_price = val_labels[-1]
    
    train_data, train_labels = utils.create_batches(train_data, train_labels, batch_size=32)
    val_data, val_labels = np.expand_dims(val_data, axis=0), np.expand_dims(val_labels, axis=0)
    
    ###########
    ## Train ##
    ###########
    if Path(f"{ticker}.best_weights.h5").is_file():
        model.load_weights(f"{ticker}.best_weights.h5")
    val_loss = NN.train_model(model,
                              train_data,
                              train_labels,
                              ticker,
                              val_data=val_data,
                              val_labels=val_labels,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              duration=60,
                              verbose=False,)
    ###############
    ## Inference ##
    ###############
    ## get latest data
    data = utils.get_stock_data(ticker, peroid, interval)
    time_series, labels = utils.make_prediction_series(data, sequence_len)
    test_data = time_series[-1:, ::]
    test_labels = labels[-1:, ::]
    test_data, test_labels = np.expand_dims(test_data, axis=0), np.expand_dims(test_labels, axis=0)
    
    model.load_weights(f"{ticker}.best_weights.h5")
    prediction = NN.infer_model(model,
                                test_data,
                                test_labels)
    
    previous_price =  yf.Ticker(ticker).history(period='1d', interval='1m')['Close'][-2]
    current_price = yf.Ticker(ticker).history().tail(1)['Close'].values[0]
    print()
    print(f"{ticker=} {previous_price=} {current_price=} {prediction.numpy()[0][0]=}")

    # Minimize our losses
    if (current_price < BUY_PRICE) and OWN:
        print(f"{utils.light_blue}SELL {BUY_PRICE=:.8} @ {current_price=:.8} (+-) {utils.light_red}{current_price - BUY_PRICE=:.8}{utils.reset}")
        OWN = False
        TOTAL += current_price - BUY_PRICE
        BUY_PRICE = 0.0
        
    tolerance_percentage = 0.001 # 0.1% tolerance
    if OWN:
        if previous_price*(1+ tolerance_percentage) < current_price:
            # Hold b/c just jumped large amount
            print(f"{utils.magenta}HOLD @ {BUY_PRICE=:.8} b/c {current_price=:.8} > {previous_price*(1+ tolerance_percentage)=:.8}{utils.reset}")
            
        elif ((prediction.numpy()[0][0] < BUY_PRICE) or
            (prediction.numpy()[0][0] < current_price)):
            # Sell if our prediction is less than our BUY_PRICE or current_price
            print(f"{utils.light_blue}SELL {BUY_PRICE=:.8} @ {current_price=:.8} (+-) {current_price - BUY_PRICE=:.8}{utils.reset}")
            OWN = False
            TOTAL += current_price - BUY_PRICE
            BUY_PRICE = 0.0
        else:
            print(f"{utils.magenta}HOLD @ {BUY_PRICE=:.8} while {current_price=:.8}{utils.reset}")
            
    elif not OWN:
        
        if prediction.numpy()[0][0] > current_price:
            # Buy if NN thinks stock will go up
            print(f"{utils.light_green}BUY @ {current_price=:.8}{utils.reset}")
            OWN = True
            BUY_PRICE = current_price
        elif previous_price*(1+ tolerance_percentage) < current_price:
            # Buy if jumped large amount
            print(f"{utils.light_green}BUY @ {current_price=:.8} b/c > {previous_price*(1+ tolerance_percentage)=:.8}{utils.reset}")
            OWN = True
            BUY_PRICE = current_price
        elif prediction.numpy()[0][0] < current_price:
            print(f"{utils.gray}PASS{utils.reset}")
            

    pct = (((start_price + TOTAL)-start_price) / abs(start_price))* 100
    if TOTAL < 0:
        print(f"{utils.light_red}{TOTAL=:.8} {pct=:.8}%{utils.reset}")
    else:
        print(f"{utils.green}{TOTAL=:.8} {pct=:.8}%{utils.reset}")
        
    if current_price - start_price < 0:
        print(f"Started trading at {start_price=:.8} Without trading bot {utils.light_red}{current_price - start_price=:.8}{utils.reset}")
    else:
        print(f"Started trading at {start_price=:.8} Without trading bot {utils.light_yellow}{current_price - start_price=:.8}{utils.reset}")
    print()
