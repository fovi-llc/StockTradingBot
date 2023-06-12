from utils import utils, NN

import numpy as np
import pandas as pd
import tensorflow as tf


#####################
## Data Collection ##
#####################
# Example usage
symbol = 'AAPL'  # Apple stock
peroid = '7d'    # max, 1m (1month), 7d (7days)
interval = '1m' # 1-minute interval
data = utils.get_stock_data(symbol, peroid, interval)

# Save Data (if necessary)
#data.to_csv("data.csv", index=False)
#data = pd.read_csv("data.csv")

num = 10
time_series, label = utils.make_time_series(data, num)

####################
## Neural Network ##
####################

input_shape = (num, 5)
batch_size = 32

model = NN.get_model(input_shape)

model.summary()

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(),)

model.fit(x=time_series,
          y=label,
          batch_size=batch_size,
          epochs=1000,
          validation_split=.1,
          shuffle=True
)
          
