from tensorflow import keras
from tensorflow.keras import layers
import json

with open("config.json", "r") as f:
    jsonFile = json.load(f)
    modelDetails = jsonFile['ModelDetails']

def getGeneratorModel():
    numberOfFeatures = modelDetails['TotalAttributes']
    attributesInput = layers.Input(shape=(numberOfFeatures,))
    attributes = layers.Dense(16 * 16 * 116)(attributesInput)
    attributes = layers.LeakyReLU()(attributes)
    attributes = layers.Reshape([16, 16, 116])(attributes)

    randomVectorSize = modelDetails['RandomVectorSize']
    randomNoiseInput = layers.Input(shape=(randomVectorSize,))
    randomNoise = layers.Dense(16 * 16 * 12)(randomNoiseInput)
    randomNoise = layers.LeakyReLU()(randomNoise)
    randomNoise = layers.Reshape([16, 16, 12])(randomNoise)

    merged = layers.Concatenate()([randomNoise, attributes])
    generator = layers.Conv2D(256, (5,5), padding="same")(merged)
    generator = layers.LeakyReLU()(generator)
    
    # 16x16 => 32x32
    generator = layers.Conv2DTranspose(256, (4,4), (2,2), padding="same")(generator)
    generator = layers.LeakyReLU()(generator)
    # 32x32 => 64x64
    generator = layers.Conv2DTranspose(256, (4,4), (2,2), padding="same")(generator)
    generator = layers.LeakyReLU()(generator)
    # 64x64 => 128x128
    generator = layers.Conv2DTranspose(256, (4,4), (2,2), padding="same")(generator)
    generator = layers.LeakyReLU()(generator)

    generator = layers.Conv2D(512, (5,5), padding="same")(generator)
    generator = layers.LeakyReLU()(generator)
    generator = layers.Conv2D(512, (5,5), padding="same")(generator)
    generator = layers.LeakyReLU()(generator)
    generator = layers.Conv2D(3, (7,7), padding="same", activation='tanh')(generator)

    generatorModel = keras.models.Model([attributesInput, randomNoiseInput], generator)

    return generatorModel