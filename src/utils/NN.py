import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization


def get_model(input_shape: list):
    model = Sequential()
    model.add(LSTM(units=128, input_shape=input_shape, return_sequences=False, kernel_initializer='glorot_uniform'))
    #model.add(BatchNormalization())
    #model.add(LSTM(units=64, return_sequences=True))
    #model.add(BatchNormalization())
    #model.add(LSTM(units=32))
    model.add(BatchNormalization())
    model.add(Dense(units=input_shape[-1]-1, activation='relu', kernel_initializer='glorot_uniform'))
    return model
