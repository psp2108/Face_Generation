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

    generatorAttributes, generatorNoise = generator.input
    generatorOutput = generator.output

    ganOutput = discriminator([generatorAttributes, generatorOutput])

    ganModel = keras.models.Model([generatorNoise, generatorAttributes], ganOutput)

    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    ganModel.compile(loss='binary_crossentropy', optimizer=opt)
    return ganModel, generator, discriminator

gan, generator, discriminator = getGanModel()
gan.summary()
plot_model(gan, to_file='gan.png', show_shapes=True, show_layer_names=True)
