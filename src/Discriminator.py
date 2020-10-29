from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

def getDiscriminatorModel():
    numberOfFeatures = 50
    attributesInput = layers.Input(shape=(numberOfFeatures,))
    attributes = layers.Dense(128 * 128 * numberOfFeatures)(attributesInput)
    attributes = layers.Reshape([128, 128, numberOfFeatures])(attributes)

    imageShape = (128, 128, 3)
    imageInput = layers.Input(shape=imageShape)

    merged = layers.Concatenate()([imageInput, attributes])
    # 128x128 => 64x64
    # 128x128 => 32x32
    discriminator = layers.Conv2D(64, (5,5), (4,4), padding='same')(merged)
    discriminator = layers.LeakyReLU(0.2)(discriminator)
    discriminator = keras.layers.Dropout(0.3)(discriminator)
    # # 64x64 => 32x32
    # discriminator = layers.Conv2D(64, (5,5), (2,2), padding='same')(discriminator)
    # discriminator = layers.LeakyReLU(0.2)(discriminator)
    # discriminator = keras.layers.Dropout(0.3)(discriminator)
    # 32x32 => 16x16
    discriminator = layers.Conv2D(64, (5,5), (2,2), padding='same')(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)
    discriminator = keras.layers.Dropout(0.3)(discriminator)
    # 16x16 => 8x8
    discriminator = layers.Conv2D(128, (5,5), (2,2), padding='same')(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)
    discriminator = keras.layers.Dropout(0.3)(discriminator)
    # 8x8 => 4x4
    discriminator = layers.Conv2D(128, (5,5), (2,2), padding='same')(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)
    discriminator = keras.layers.Dropout(0.3)(discriminator)
    # 4x4 => 16 x neurons
    discriminator = keras.layers.Flatten()(discriminator)
    discriminator = layers.Dense(1, activation='sigmoid')(discriminator)

    discriminatorModel = keras.models.Model([attributesInput, imageInput], discriminator)

    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    discriminatorModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return discriminatorModel

