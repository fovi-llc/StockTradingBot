import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization




def get_model(input_shape: list):
    model = Sequential()
    model.add(LSTM(units=128, input_shape=input_shape, return_sequences=False))
    #model.add(BatchNormalization())
    #model.add(LSTM(units=64, return_sequences=True))
    #model.add(BatchNormalization())
    #model.add(LSTM(units=32))
    model.add(BatchNormalization())
    model.add(Dense(units=5, activation='relu'))
    return model
