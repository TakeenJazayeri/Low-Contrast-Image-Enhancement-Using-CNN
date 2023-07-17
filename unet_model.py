import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_Unet_mdel (row, col):
    # Feature Extraction Step
    inputs = layers.Input(shape=(row, col, 3))

    feature1 = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same')(inputs)
    x1 = layers.MaxPool2D(2)(feature1)

    feature2 = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same')(x1)
    x2 = layers.MaxPool2D(2)(feature2)

    feature3 = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same')(x2)
    x3 = layers.MaxPool2D(2)(feature3)

    x4 = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding='same')(x3)
    x5 = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding='same')(x4)

    # Probability Reconstraction Step
    x6 = layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", padding='same')(x5)
    x6 = layers.UpSampling2D((2, 2), interpolation='bilinear')(x6)
    x6 = layers.concatenate([x6, feature3])
    
    x7 = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", padding='same')(x6)
    x7 = layers.UpSampling2D((2, 2), interpolation='bilinear')(x7)
    x7 = layers.concatenate([x7, feature2])
    
    x8 = layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu", padding='same')(x7)
    x8 = layers.UpSampling2D((2, 2), interpolation='bilinear')(x8)
    x8 = layers.concatenate([x8, feature1])

    outputs = layers.Conv2D(filters=1, kernel_size=3, activation="relu", padding='same')(x8)
    return keras.Model(inputs, outputs)