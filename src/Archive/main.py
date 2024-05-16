import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import yfinance as yf

from utils import utils, NN, upward_trend, resistance_line

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf


################
## Parameters ##
################
sequence_len = 10
input_shape = (sequence_len, 4)
batch_size = 16
lr = 0.001

peroid = '360m' #1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
interval = '2m' # 1m, 2m, 5m, 15m, 30m, 60m 90m, 1h, 1d, 5d, 1wk, 1mo
ticker = "RTX"

#######################
## Find Upward Trend ##
#######################
#stocks = upward_trend.find_upward()
#ticker = next(iter(stocks))
#exit()

#####################
## Resistance_line ##
#####################
#peroid = '75m' #1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
#interval = '1m' # 1m, 2m, 5m, 15m, 30m, 60m 90m, 1h, 1d, 5d, 1wk, 1mo
#prices, volume, RSIs, macd, bollinger_bands = resistance_line.above_or_below_resistance_line(ticker, peroid, interval)
#print(f"prices")
#print(f"{volume=}")
#print(f"{RSIs=}")
#print(f"{macd=}")
#print(f"{bollinger_bands=}")
#resistance_line.plot_metrics_with_resistance(ticker, peroid, interval)


####################
## Neural Network ##
####################
#loss_fn = tf.keras.losses.MeanSquaredError()
loss_fn = tf.keras.losses.MeanAbsoluteError()
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

    ##########################
    ## Plot Resistance_line ##
    ##########################
    #resistance_line.plot_metrics_with_resistance(ticker, peroid="120m", interval="1m")
    
    ##########
    ## Data ##
    ##########
    data = utils.get_stock_data(ticker, peroid, interval)
    time_series, labels = utils.make_time_series(data, sequence_len)
   
    train_data = time_series[:, ::]
    train_labels = labels[:, ::]
    
    val_data = time_series[-1:, ::]
    val_labels = labels[-1:, ::]

    prev_price = val_labels[-1]
    
    train_data, train_labels = utils.create_batches(train_data, train_labels, batch_size=32)
    #train_data, train_labels = utils.create_batches(val_data, val_labels, batch_size=32)
    val_data, val_labels = np.expand_dims(val_data, axis=0), np.expand_dims(val_labels, axis=0)
    
    ###########
    ## Train ##
    ###########
    if Path(f".weights/{ticker}.best_weights.h5").is_file():
        model.load_weights(f".weights/{ticker}.best_weights.h5")
    val_loss = NN.train_model(model,
                              train_data,
                              train_labels,
                              ticker,
                              val_data=val_data,
                              val_labels=val_labels,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              duration=15,
                              verbose=False,)
    
    ###############
    ## Inference ##
    ###############
    ## get latest data
    data = utils.get_stock_data(ticker, peroid=peroid, interval="1m")
    time_series, labels = utils.make_prediction_series(data, sequence_len)
    test_data = time_series[-1:, ::]
    test_labels = labels[-1:, ::]
    test_data, test_labels = np.expand_dims(test_data, axis=0), np.expand_dims(test_labels, axis=0)
    
    model.load_weights(f".weights/{ticker}.best_weights.h5")
    prediction = NN.infer_model(model,
                                test_data,
                                test_labels)
    
    previous_price =  yf.Ticker(ticker).history(period='1d', interval='1m')['Close'][-2]
    current_price = yf.Ticker(ticker).history().tail(1)['Close'].values[0]
    print()
    print(f"{ticker=} {previous_price=} {current_price=} {prediction.numpy()[0][0]=}")
        
    tolerance_percentage = 0.001 # 0.1% tolerance
    tolerance_error = 0.10 # 0.10 cents
    val_loss = val_loss / 2
    if OWN:
        if previous_price*(1 + tolerance_percentage) < current_price:
            # Hold b/c just jumped large amount
            print(f"{utils.magenta}HOLD @ {BUY_PRICE=:.8} b/c {current_price=:.8} > {previous_price*(1 + tolerance_percentage)=:.8}{utils.reset}")

        elif (current_price < BUY_PRICE):
            # minimize our LOSS
            print(f"{utils.light_blue}SELL @ {current_price=:.8} b/c < {BUY_PRICE=:.8} (-) {utils.light_red}{current_price - BUY_PRICE=:.8}{utils.reset}")
            OWN = False
            TOTAL += current_price - BUY_PRICE
            BUY_PRICE = 0.0
            
        elif previous_price > current_price:
            # Sell b/c price dropped
            print(f"{utils.light_blue}SELL @ {current_price=:.8} b/c < {previous_price=:.8} (+-) {current_price - BUY_PRICE=:.8}{utils.reset}")
            OWN = False
            TOTAL += current_price - BUY_PRICE
            BUY_PRICE = 0.0

        # We dont want to act on our network if we make a bad prediction. Continue if MAE is > 5 cents
        elif val_loss <= tolerance_error:
        
            if prediction.numpy()[0][0] - val_loss < BUY_PRICE:
                # Sell if our prediction is less than our BUY_PRICE
                print(f"{utils.light_blue}SELL @ {current_price=:.8} b/c {prediction.numpy()[0][0]-val_loss=:.8} < {BUY_PRICE=:.8} @ (+-) {current_price - BUY_PRICE=:.8}{utils.reset}")
                OWN = False
                TOTAL += current_price - BUY_PRICE
                BUY_PRICE = 0.0
            
            elif (prediction.numpy()[0][0] - val_loss < current_price):
                # Sell if our prediction is less than our current_price
                print(f"{utils.light_blue}SELL @ {current_price=:.8} b/c {prediction.numpy()[0][0]-val_loss=:.8} < {current_price=:.8} @ (+-) {current_price - BUY_PRICE=:.8}{utils.reset}")
                OWN = False
                TOTAL += current_price - BUY_PRICE
                BUY_PRICE = 0.0
            
            elif prediction.numpy()[0][0] - val_loss >= current_price:
                # Hold if NN thinks price will go up
                print(f"{utils.magenta}HOLD @ {BUY_PRICE=:.8} b/c {current_price=:.8} > {prediction.numpy()[0][0]-val_loss=:.8}{utils.reset} ")
        
        else:
            # Sell if val_loss <= tolerance_error
            print(f"{utils.light_blue}SELL @ {current_price=:.8} b/c {prediction.numpy()[0][0]-val_loss=:.8} & {val_loss <= tolerance_error=} (+-) {current_price - BUY_PRICE=:.8}{utils.reset}")
            OWN = False
            TOTAL += current_price - BUY_PRICE
            BUY_PRICE = 0.0
            
    elif not OWN:
        if previous_price*(1 - tolerance_percentage) > current_price:
            # Pass b/c price dropped a lot
            print(f"{utils.gray}PASS  b/c {current_price=:.8} < {previous_price*(1 - tolerance_percentage)=:.8}{utils.reset}")
            
        elif previous_price*(1+ tolerance_percentage) < current_price:
            # Buy if jumped large amount
            print(f"{utils.light_green}BUY @ {current_price=:.8} b/c > {previous_price*(1+ tolerance_percentage)=:.8}{utils.reset}")
            OWN = True
            BUY_PRICE = current_price

        # We dont want to act on our network if we make a bad prediction. Continue if MAE is > 5 cents
        elif val_loss <= tolerance_error:
        
            if prediction.numpy()[0][0] - val_loss > current_price:
                # Buy if NN thinks stock will go up
                print(f"{utils.light_green}BUY @ {current_price=:.8} b/c < {prediction.numpy()[0][0]-val_loss=:.8}{utils.reset}")
                OWN = True
                BUY_PRICE = current_price
                
            elif prediction.numpy()[0][0] - val_loss < current_price:
                print(f"{utils.gray}PASS b/c {prediction.numpy()[0][0]-val_loss=:.8} < {current_price=:.8}{utils.reset}")

        else:
            print(f"{utils.gray}PASS b/c {val_loss <= tolerance_error=} {prediction.numpy()[0][0]-val_loss=:.8} < {current_price=:.8}{utils.reset}")
            

    pct = (((start_price + TOTAL)-start_price) / abs(start_price))* 100
    if TOTAL < 0:
        print(f"{utils.light_red}{TOTAL=:.8} {pct=:.8}%{utils.reset}")
    else:
        print(f"{utils.green}{TOTAL=:.8} {pct=:.8}%{utils.reset}")
        
    if current_price - start_price < 0:
        print(f"Started trading @ {start_price=:.8} Without trading bot {utils.light_red}{current_price - start_price=:.8}{utils.reset}")
    else:
        print(f"Started trading @ {start_price=:.8} Without trading bot {utils.light_yellow}{current_price - start_price=:.8}{utils.reset}")
    print()
