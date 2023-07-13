import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import yfinance as yf

from utils import utils, NN


################
## Parameters ##
################
sequence_len = 10
input_shape = (sequence_len, 4)
batch_size = 32
lr = 0.001

peroid = '1d'    # max, 1mo (1month), 7d (7days)
interval = '1m' # 1-minute interva
ticker = "GOOGL"

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
    
    val_data = time_series[-1, ::]
    val_labels = labels[-1, ::]

    prev_price = val_labels[-1]
    
    train_data, train_labels = utils.create_batches(train_data, train_labels, batch_size=32)
    val_data, val_labels = np.expand_dims(np.expand_dims(val_data, axis=0), axis=0), np.expand_dims(np.expand_dims(val_labels, axis=0), axis=0)

    ###########
    ## Train ##
    ###########
    model.load_weights("best_weights.h5")
    NN.train_model(model,
                   train_data,
                   train_labels,
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
    test_data = time_series[-1, ::]
    test_labels = labels[-1, ::]
    test_data, test_labels = np.expand_dims(np.expand_dims(test_data, axis=0), axis=0), np.expand_dims(np.expand_dims(test_labels, axis=0), axis=0)
    
    model.load_weights("best_weights.h5")
    prediction = NN.infer_model(model,
                                test_data,
                                test_labels)
    
    current_price = yf.Ticker(ticker).history().tail(1)['Close'].values[0]
    print()
    print(f"{current_price=} {prediction.numpy()[0][0]=}")

    if (current_price < BUY_PRICE) and OWN:
        print(f"{utils.light_blue}SELL {BUY_PRICE=:.8} @ {current_price=:.8} (+-) {current_price - BUY_PRICE=:.8}{utils.reset}")
        OWN = False
        TOTAL += current_price - BUY_PRICE
        BUY_PRICE = 0.0
    if OWN:
        if (prediction.numpy()[0][0] < BUY_PRICE or
            prediction.numpy()[0][0] < current_price):
            print(f"{utils.light_blue}SELL {BUY_PRICE=:.8} @ {current_price=:.8} (+-) {current_price - BUY_PRICE=:.8}{utils.reset}")
            OWN = False
            TOTAL += current_price - BUY_PRICE
            BUY_PRICE = 0.0
        else:
            print(f"{utils.magenta}HOLD @ {BUY_PRICE=:.8} while {current_price=:.8}{utils.reset}")
    elif not OWN:
        if prediction.numpy()[0][0] > current_price:
            print(f"{utils.light_green}BUY @ {current_price=:.8}{utils.reset}")
            OWN = True
            BUY_PRICE = current_price
        elif prediction.numpy()[0][0] < current_price:
            print(f"{utils.gray}PASS{utils.reset}")
            

    if TOTAL < 0:
        print(f"{utils.light_red}{TOTAL=}{utils.reset}")
    else:
        print(f"{utils.green}{TOTAL=}{utils.reset}")
    print()
