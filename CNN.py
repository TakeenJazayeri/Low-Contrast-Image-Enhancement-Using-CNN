import numpy as np
import os, cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from transformation import rotate_10_right, rotate_10_left, flip
from random import sample
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X, Y = [], []
row, col = 512, 512

with os.scandir('Dataset\\Result') as imagefiles:
    for imagefile in imagefiles:
        image = cv2.imread(f'Dataset\\Main\\{imagefile.name}', cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (col, row))
        X.append(resized_image.reshape(row, col, 3))
        X.append(rotate_10_right(resized_image).reshape(row, col, 3))
        X.append(rotate_10_left(resized_image).reshape(row, col, 3))
        X.append(flip(resized_image).reshape(row, col, 3))

        prob_map = cv2.imread(f'Dataset\\Result\\{imagefile.name}', 0)
        resized_prob_map = cv2.resize(prob_map, (col, row))
        Y.append(resized_prob_map.reshape(row, col, 1))
        Y.append(rotate_10_right(resized_prob_map).reshape(row, col, 1))
        Y.append(rotate_10_left(resized_prob_map).reshape(row, col, 1))
        Y.append(flip(resized_prob_map).reshape(row, col, 1))

train_perc = 0.8
selected_index = sample(list(range(1, len(X))), int(len(X)*train_perc))

X_train, X_val, Y_train, Y_val = [], [], [], []
for i in range(len(X)):
    if i in selected_index:
        X_train.append(X[i])
        Y_train.append(Y[i])
    else:
        X_val.append(X[i])
        Y_val.append(Y[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_val = np.array(X_val)
Y_val = np.array(Y_val)


model = keras.Sequential([
    # Feature Extraction Step
    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same', input_shape=[row, col, 3]),
    layers.MaxPool2D(2),
    
    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(2),

    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(2),

    layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding='same'),

    # Probability Reconstraction Step
    layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.UpSampling2D((2, 2), interpolation='bilinear'),

    layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.UpSampling2D((2, 2), interpolation='bilinear'),

    layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu", padding='same'),
    layers.UpSampling2D((2, 2), interpolation='bilinear'),

    layers.Conv2D(filters=1, kernel_size=3, activation="relu", padding='same'),
])

model.compile(optimizer = 'adam', loss = keras.losses.MeanSquaredError())

history = model.fit(
    X_train, Y_train, validation_data=(X_val, Y_val),
    batch_size=5, epochs=10, verbose=1
)

model.save('cnn_model')