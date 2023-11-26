import tensorflow as tf
import numpy as np


actions = np.array(['hello', 'thankyou', 'love', 'friend', 'good', 'meet', 'you', 'think', 'bed', 'have'])

model = tf.keras.Sequential([

    tf.keras.layers.GRU(units=128, input_shape=(30, 1662), return_sequences=True),

    tf.keras.layers.GRU(units=64, return_sequences=True),

    tf.keras.layers.GRU(units=32, return_sequences=True),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128),

    tf.keras.layers.Dense(64),

    tf.keras.layers.Dense(10, activation='softmax') 
])

# Wczytaj wagi do nowego modelu
model.load_weights('gru_weights.h5')