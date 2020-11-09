from tensorflow import keras
from tensorflow.keras import layers
import json

with open("config.json", "r") as f:
    jsonFile = json.load(f)
    modelDetails = jsonFile['ModelDetails']

def getGeneratorModel():
    numberOfFeatures = modelDetails['TotalAttributes']
    attributesInput = layers.Input(shape=(numberOfFeatures,))
    attributes = layers.Dense(4 * 4 * numberOfFeatures)(attributesInput)
    attributes = layers.Reshape([4, 4, numberOfFeatures])(attributes)

    randomVectorSize = modelDetails['RandomVectorSize']
    randomNoiseInput = layers.Input(shape=(randomVectorSize,))
    randomNoise = layers.Dense(4 * 4 * 256)(randomNoiseInput)
    randomNoise = layers.Reshape([4, 4, 256])(randomNoise)

    merged = layers.Concatenate()([randomNoise, attributes])
    generator = layers.BatchNormalization()(merged)
    # 4x4 => 8x8
    generator = layers.Conv2DTranspose(128, (5,5), (2,2), padding="same")(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    # 8x8 => 16x16
    generator = layers.Conv2DTranspose(128, (5,5), (2,2), padding="same")(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    # 16x16 => 32x32
    generator = layers.Conv2DTranspose(128, (5,5), (2,2), padding="same")(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    # 32x32 => 64x64
    generator = layers.Conv2DTranspose(128, (5,5), (2,2), padding="same")(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    # 64x64 => 128x128
    generator = layers.Conv2DTranspose(3, (5,5), (2,2), padding="same", activation="selu")(generator)

    generatorModel = keras.models.Model([attributesInput, randomNoiseInput], generator)

    return generatorModel