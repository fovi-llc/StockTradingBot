import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import utils, NN

################
## Parameters ##
################
sequence_len = 10
input_shape = (sequence_len, 4)
epochs = 1000
batch_size = 32
lr = 0.001

#####################
## Data Collection ##
#####################
peroid = '1d'    # max, 1mo (1month), 7d (7days)
interval = '1m' # 1-minute interval
#data1 = utils.get_stock_data("AAPL", peroid, interval)
data = utils.get_stock_data("GOOGL", peroid, interval)
#data3 = utils.get_stock_data("NVDA", peroid, interval)
#data4 = utils.get_stock_data("SPY", peroid, interval)
#data5 = utils.get_stock_data("VOO", peroid, interval)
#data6 = utils.get_stock_data("AMD", peroid, interval)

#data = pd.concat([data1, data2, data3, data4, data5, data6], axis=0)
#data = pd.concat([data1, data2], axis=0)

# Save Data (if necessary)
data.to_csv("data.csv", index=False)
data = pd.read_csv("data.csv")

time_series, labels = utils.make_time_series(data, sequence_len)

diff = []
for i in range(1, len(labels)):
    diff.append(abs(labels[i-1] - labels[i]))
    print(labels[i-1] - labels[i])
print(f"{np.average(diff)=}")
print(time_series.shape, labels.shape)

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
num_sim = 20
pred = []
labe = []


OWN = False
BUT_PRICE = 0.0
for i in range(num_sim, 0, -1):

    ##########
    ## Data ##
    ##########
    train_data = time_series[:-i, ::]
    train_labels = labels[:-i, ::]

    val_data = time_series[-i-2:-i, ::]
    val_labels = labels[-i-2:-i, ::]

    test_data = time_series[-i, ::]
    test_labels = labels[-i, ::]

    prev_price = val_labels[-1]
    

    train_data, train_labels = utils.create_batches(train_data, train_labels, batch_size=32)
    val_data, val_labels = np.expand_dims(val_data, axis=0), np.expand_dims(val_labels, axis=0)
    test_data, test_labels = np.expand_dims(np.expand_dims(test_data, axis=0), axis=0), np.expand_dims(np.expand_dims(test_labels, axis=0), axis=0)

    ###########
    ## Train ##
    ###########
    model.load_weights("best_weights.h5")
    NN.train_model(model,
                   train_data,
                   train_labels,
                   val_data=val_data,
                   val_labels=val_labels,
                   epochs=epochs,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   duration=55)

    ###############
    ## Inference ##
    ###############
    model.load_weights("best_weights.h5")
    prediction = NN.infer_model(model,
                                test_data,
                                test_labels)
    print()
    print(f"{prev_price[0]=}")
    print(f"{test_labels[0][0][0]=}")
    print(f"{prediction.numpy()[0][0]=}")
    labe.append(test_labels[0][0][0])
    pred.append(prediction.numpy()[0][0])
    

    if prediction.numpy()[0][0] > prev_price[0]:
        if not OWN:
            print("BUY")
        else:
            print("HOLD")
    elif prediction.numpy()[0][0] < prev_price[0]:
        if not OWN:
            print("PASS")
        else:
            print("SELL")
    print()
    
for i, j in zip(labe, pred):
    print(f"label={i:.4f}, pred={j:.4f}, abs_diff={abs(i-j):.4f}, diff={i-j:.4f}")

