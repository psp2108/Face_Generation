from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from src.Discriminator import getDiscriminatorModel
from src.Generator import getGeneratorModel

def getGanModel():
    generator = getGeneratorModel()
    discriminator = getDiscriminatorModel()

    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    # discriminatorModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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
    folder = 'Model Diagrams/'
    plot_model(gan, to_file=folder+'gan.png', show_shapes=True, show_layer_names=True)
    plot_model(generator, to_file=folder+'generator.png', show_shapes=True, show_layer_names=True)
    plot_model(discriminator, to_file=folder+'discriminator.png', show_shapes=True, show_layer_names=True)

# main()