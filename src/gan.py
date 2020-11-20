from tensorflow import keras
from tensorflow.keras.utils import plot_model
from Discriminator import getDiscriminatorModel
from Generator import getGeneratorModel
import sys
import json
import os

def getGanModel():
    generator = getGeneratorModel()
    discriminator = getDiscriminatorModel()

    # RMSprop -> Gradient based optimization technique
    opt = keras.optimizers.RMSprop(lr=0.0001)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    discriminator.trainable = False

    generatorAttributes, generatorNoise = generator.input
    generatorOutput = generator.output

    ganOutput = discriminator([generatorAttributes, generatorOutput])

    ganModel = keras.models.Model([generatorNoise, generatorAttributes], ganOutput)

    # RMSprop -> Gradient based optimization technique
    opt = keras.optimizers.RMSprop(lr=0.0001)
    ganModel.compile(loss='binary_crossentropy', optimizer=opt)
    return ganModel, generator, discriminator

def main():
    gan, generator, discriminator = getGanModel()
    gan.summary()

    with open("config.json", "r") as f:
        jsonFile = json.load(f)
        modelDetails = jsonFile['ModelDetails']

    folder = modelDetails['ModelDiagrams']

    if not os.path.isdir(folder):
        os.makedirs(folder)

    plot_model(gan, to_file=os.path.join(folder, 'gan.png'), show_shapes=True, show_layer_names=True)
    plot_model(generator, to_file=os.path.join(folder, 'generator.png'), show_shapes=True, show_layer_names=True)
    plot_model(discriminator, to_file=os.path.join(folder, 'discriminator.png'), show_shapes=True, show_layer_names=True)

if __name__ == "__main__":
    main()