from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from Discriminator import getDiscriminatorModel
from Generator import getGeneratorModel

def getGanModel():
    generator = getGeneratorModel()
    discriminator = getDiscriminatorModel()

    discriminator.trainable = False

    generatorNoise, generatorAttributes = generator.input
    generatorOutput = generator.output

    ganOutput = discriminator([generatorOutput, generatorAttributes])

    ganModel = keras.models.Model([generatorNoise, generatorAttributes], ganOutput)

    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    ganModel.compile(loss='binary_crossentropy', optimizer=opt)
    return ganModel

gan = getGanModel()
