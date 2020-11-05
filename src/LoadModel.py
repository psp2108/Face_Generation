from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from PIL import Image
import numpy as np
from gan import getGanModel

import cv2
import pandas as pd
import os
from tqdm import tqdm

gan, generator, discriminator = getGanModel()

def plotSaveImage(image, savePath = ''):
    data = (image.numpy() * 255)[0]
    plt.imshow(data, cmap=plt.cm.viridis)
    plt.show()
    if len(savePath):
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(savePath)

generator.load_weights("P:\\GAN Learning\\Face_Generation\\models\\generator12449.h5")

# Test Generator image

features = np.array([[-1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,-1,1,1,-1,1]])
features = np.asarray(features).astype('float32')
features += 0.05 * np.random.random(features.shape)

# features = tf.random.normal(shape=[1, 40])
randomNoise = tf.random.normal(shape=[1, 100])
# gImage = generator([np.array([features,features,features]), np.array([randomNoise,randomNoise,randomNoise])], training=False)
gImage = generator([features, randomNoise], training=False)
print(gImage.shape)
plotSaveImage(gImage, "P:\\GAN Learning\\Face_Generation\\models\\generator10249-sample.png")