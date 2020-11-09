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
import json
from tqdm import tqdm

def plotSaveImage(image, savePath = ''):
    data = (image.numpy() * 255)[0]
    data = data.astype('uint8')
    plt.imshow(data, cmap=plt.cm.viridis)
    plt.show()
    if len(savePath):
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(savePath)

with open("config.json", "r") as f:
    jsonFile = json.load(f)
    modelDetails = jsonFile['ModelDetails']
modelPath = os.path.join(modelDetails['TrainedModel'], 'generator_latest.h5')

generator = keras.models.load_model(modelPath)

# Test Generator image
features = np.array([modelDetails['Testing']['Attributes']])
features = np.asarray(features).astype('float32')
features += 0.05 * np.random.random(features.shape)

randomNoise = tf.random.normal(shape=[1, modelDetails['RandomVectorSize']])
gImage = generator([features, randomNoise], training=False)
print(gImage.shape)
plotSaveImage(gImage, os.path.join(modelDetails['Testing']['OutputFolder'], modelDetails['Testing']['OutputImage']))