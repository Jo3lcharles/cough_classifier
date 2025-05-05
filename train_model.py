import tensorflow as tf
import numpy as np
import os

# Load preprocessed data (mock data for testing)
def load_data(data_dir):
    # Generate fake MFCC data (replace with real data later)
    X_normal = np.random.rand(100, 13)  # 100 normal coughs (13 MFCC features each)
    X_abnormal = np.random.rand(100, 13)  # 100 abnormal coughs
    X = np.vstack([X_normal, X_abnormal])
    y = np.array([0]*100 + [1]*100)  # 0=normal, 1=abnormal
    return X, y

X_train, y_train = load_data("data/train")
X_test, y_test = load_data("data/test")

# Lite model (optimized for edge devices)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(13,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save model
model.save("models/cough_model.h5")

# Convert to TFLite (for Raspberry Pi)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("models/cough_model.tflite", "wb") as f:
    f.write(tflite_model)