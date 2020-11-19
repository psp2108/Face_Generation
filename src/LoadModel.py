import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from PIL import Image
import numpy as np

import json


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
    csvDetails = jsonFile['CSVDetails']
modelPath = os.path.join(modelDetails['ModelRootFolder'], modelDetails['TrainedModel'], "generator_{}.h5".format(modelDetails['Testing']['Version'] or "latest"))

generator = keras.models.load_model(modelPath)

# Test Generator
features = np.array([modelDetails['Testing']['Attributes']])
features = np.asarray(features).astype('float32')
features += 0.05 * np.random.random(features.shape)

randomNoise = tf.random.normal(shape=[1, modelDetails['RandomVectorSize']])

gImage = generator([features, randomNoise], training=False)

temp = open(os.path.join(csvDetails["CSVRootPath"], csvDetails["CombinedCSV"]), "r")
attributesList = temp.readline().replace("\n","").split(",")[1:]
temp.close()

maxLen = 20

print("-"*100)
print("Attributes passed")
print("-"*100)
for i in range(modelDetails['TotalAttributes']):
    print(attributesList[i], " " * (maxLen - len(attributesList[i])), modelDetails['Testing']["Attributes"][i])
print("-"*100)
print("Random Latent: ")
for i in np.array(randomNoise[0]):
    print(i, end="  ")
print()
print("-"*100)

print("Shape of image(s) generated:", gImage.shape)
plotSaveImage(gImage, os.path.join(modelDetails['Testing']['OutputFolder'], modelDetails['Testing']['OutputImage']))