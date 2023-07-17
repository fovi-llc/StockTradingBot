import time
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Attention
from tensorflow.keras import regularizers

class Attention(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='random_normal', trainable=True, name="attention_w")
        self.b = self.add_weight(shape=(input_shape[1],), initializer='zeros', trainable=True, name="attention_b")

    def call(self, inputs):
        e = tf.matmul(inputs, self.W) + self.b
        a = tf.nn.softmax(e, axis=1)
        weighted_sum = tf.reduce_sum(inputs * a, axis=1)
        return weighted_sum

def get_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    lstm = tf.keras.layers.LSTM(units=input_shape[0], return_sequences=True)(inputs)
    attention = Attention(input_shape)(lstm)
    bn1 = BatchNormalization()(attention)
    dense = tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(bn1)
    bn2 = BatchNormalization()(dense)
    outputs = tf.keras.layers.Dense(units=1)(bn2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_model(model, train_data, train_labels, val_data=None, val_labels=None, loss_fn=None, optimizer=None, duration=55, verbose=True):

    start_time = time.time()

    best_val_loss = float('inf')
    while time.time() - start_time < duration:

        # condition for early stoppage
        if best_val_loss < 0.005: break
        
        # Train Loop
        epoch_loss = 0.0
        for batch, (inputs, labels) in enumerate(zip(train_data, train_labels)):
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
    
                batch_loss = loss_fn(labels, predictions)
    
            gradients = tape.gradient(batch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
            epoch_loss += batch_loss

        avg_train_loss = epoch_loss / (batch + 1)
    
        # Validation Loop
        if val_data.all():
            val_loss = 0.0
            for batch, (inputs, labels) in enumerate(zip(val_data, val_labels)):
                with tf.GradientTape() as tape:
                    predictions = model(inputs, training=False)
            
                    batch_loss = loss_fn(labels, predictions)
            
                val_loss += batch_loss
            
            avg_val_loss = val_loss / (batch + 1)
            print_ln = f"Time {datetime.now().time()}, Run_Time {time.time() - start_time:.4} Loss: {avg_train_loss:.4f} Val_Loss: {avg_val_loss:.4f}"
            
            if avg_val_loss < best_val_loss:
                print_ln += f"\nVal_Loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}"
                best_val_loss = avg_val_loss
                model.save_weights("best_weights.h5")
            else:
                print_ln += f"\nVal_Loss did not improve from {best_val_loss:.4f}"
        else:
            print_ln = f"Time {datetime.now().time()}, Run_Time {time.time() - start_time}, Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}"
            if avg_train_loss < best_val_loss:
                #print(f"Loss improved from {best_val_loss:.4f} to {avg_train_loss:.4f}", end="\r")
                print_ln += f"\nLoss improved from {best_val_loss:<5.4f} to {avg_train_loss:<5.4f}"
                best_val_loss = avg_train_loss
                model.save_weights("best_weights.h5")
            else:
                print_ln += f"\nTrain_Loss did not improve from {best_val_loss:.4f}"

        
        if verbose:
            print(print_ln)
            print("\033[2A", end="")  # Move the cursor 2 lines up
            print("\033[K", end="")   # Clear the current line
        else:
            print("                                                                          ", end="\r")
            print(f"Time {datetime.now().time()}, Run_Time {time.time() - start_time:.4} {best_val_loss=:.4f}", end="\r")

def infer_model(model, test_data, test_labels):
    
    for batch, (inputs, labels) in enumerate(zip(test_data, test_labels)):
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=False)
    
    return predictions
