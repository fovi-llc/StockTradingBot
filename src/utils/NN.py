import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization




def get_model(input_shape: list):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='relu'))
    return model
