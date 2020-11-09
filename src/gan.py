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

    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer="rmsprop")
    discriminator.trainable = False

    generatorAttributes, generatorNoise = generator.input
    generatorOutput = generator.output

    ganOutput = discriminator([generatorAttributes, generatorOutput])

    ganModel = keras.models.Model([generatorNoise, generatorAttributes], ganOutput)

    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
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

if len(sys.argv) == 1:
    print("Nothing Happened! execute 'python gan.py -save-diagram' to save the model diagrams")
elif len(sys.argv) == 2:
    if sys.argv[1] == '-save-diagram':
        main()
    else:
        print("Invalid command, type '-save-diagram' to save the model diagrams")
else:
    print("Extra parameters supplied")