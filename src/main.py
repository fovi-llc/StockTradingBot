from utils import utils, NN

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


#####################
## Data Collection ##
#####################
symbol = 'AAPL'  # Apple stock
peroid = '7d'    # max, 1m (1month), 7d (7days)
interval = '1m' # 1-minute interval
data = utils.get_stock_data(symbol, peroid, interval)

# Save Data (if necessary)
data.to_csv("data.csv", index=False)
data = pd.read_csv("data.csv")

num_in_sequence = 60
time_series, label = utils.make_time_series(data, num_in_sequence)

####################
## Neural Network ##
####################

input_shape = (num_in_sequence, 5)
batch_size = 32

lr = 0.1
loss = tf.keras.losses.MeanSquaredError()
#loss = tf.keras.losses.MeanAbsoluteError()
#loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

model = NN.get_model(input_shape)
model.summary()

model.compile(loss=loss,
              optimizer=optimizer,)

checkpoint = ModelCheckpoint('best_weights.h5',
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1)
#model.load_weights("best_weights.h5")
#model.load_weights("final_weights.h5")


###############
## Inference ##
###############
#predictions = model.predict(np.array([time_series[0]]))
#print(predictions)
#print(label[0])
#print(loss(label[0], predictions[0]))
#exit()

###########
## Train ##
###########
model.fit(x=time_series,
          y=label,
          batch_size=batch_size,
          epochs=1000,
          validation_split=.1,
          shuffle=True,
          callbacks=[checkpoint]
)

model.save_weights("final_weights.h5")
