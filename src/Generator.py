from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

def getGeneratorModel():
    numberOfFeatures = 40
    attributesInput = layers.Input(shape=(numberOfFeatures,))
    attributes = layers.Dense(4 * 4 * numberOfFeatures)(attributesInput)
    attributes = layers.Reshape([4, 4, numberOfFeatures])(attributes)

    randomVectorSize = 100
    randomNoiseInput = layers.Input(shape=(randomVectorSize,))
    randomNoise = layers.Dense(4 * 4 * 256)(randomNoiseInput)
    # randomNoise = layers.LeakyReLU(alpha=0.2)(randomNoise)
    randomNoise = layers.Reshape([4, 4, 256])(randomNoise)

    merged = layers.Concatenate()([randomNoise, attributes])
    generator = layers.BatchNormalization()(merged)
    # 4x4 => 8x8
    generator = layers.Conv2DTranspose(128, (5,5), (2,2), padding="same", activation="selu")(generator)
    generator = layers.BatchNormalization()(generator)
    # generator = layers.LeakyReLU(alpha=0.2)(generator)
    # 8x8 => 16x16
    generator = layers.Conv2DTranspose(128, (5,5), (2,2), padding="same", activation="selu")(generator)
    generator = layers.BatchNormalization()(generator)
    # generator = layers.LeakyReLU(alpha=0.2)(generator)
    # 16x16 => 32x32
    generator = layers.Conv2DTranspose(128, (5,5), (2,2), padding="same", activation="selu")(generator)
    generator = layers.BatchNormalization()(generator)
    # generator = layers.LeakyReLU(alpha=0.2)(generator)
    # 32x32 => 64x64
    generator = layers.Conv2DTranspose(128, (5,5), (2,2), padding="same", activation="selu")(generator)
    generator = layers.BatchNormalization()(generator)
    # generator = layers.LeakyReLU(alpha=0.2)(generator)
    # 64x64 => 128x128
    # ------------------------------ This ------------------------------
    generator = layers.Conv2DTranspose(3, (5,5), (2,2), padding="same", activation="selu")(generator)
    # ---------------------------- or this -----------------------------
    # generator = layers.Conv2DTranspose(128, (5,5), (2,2), padding="same", activation="selu")(generator)
    # generator = layers.BatchNormalization()(generator)
    # # generator = layers.LeakyReLU(alpha=0.2)(generator)

    # generator = layers.Conv2D(3, (7,7), activation='tanh', padding='same')(generator)
    # ---------------------------- or this -----------------------------

    generatorModel = keras.models.Model([attributesInput, randomNoiseInput], generator)

    return generatorModel