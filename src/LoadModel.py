import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from PIL import Image
import numpy as np

import json

class generatorModule():

    def __init__(self):
        with open("config.json", "r") as f:
            jsonFile = json.load(f)
            self.modelDetails = jsonFile['ModelDetails']
            self.csvDetails = jsonFile['CSVDetails']

        self.modelPath = os.path.join(self.modelDetails['ModelRootFolder'], self.modelDetails['TrainedModel'], "generator_{}.h5")
        self.modelVersion = self.modelDetails['Testing']['Version'] or "latest"

        self.outputFolder = self.modelDetails['Testing']['OutputFolder']
        self.outputImage = self.modelDetails['Testing']['OutputImage']
        self.outputImagePath = os.path.join(self.outputFolder, self.outputImage)
        
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)

        self.randomVectorSize = self.modelDetails['RandomVectorSize']
        self.featureVectorSize = self.modelDetails['TotalAttributes']
        self.dummyAttributes = self.modelDetails['Testing']['Attributes']

        temp = open(os.path.join(self.csvDetails["CSVRootPath"], self.csvDetails["CombinedCSV"]), "r")
        self.attributesListOrder = temp.readline().lower().replace("\n","").split(",")[1:]
        temp.close()

        self.loadGenerator()

    def setOutputFolder(self, outputFolder):
        self.outputFolder = outputFolder

    def getAttributesSize(self):
        return self.featureVectorSize

    def getRandomNoiseSize(self):
        return self.randomVectorSize

    def getDummyAttributes(self):
        return self.dummyAttributes

    def getFeatureVectorFromDict(self, args):
        if (type(args) == type(list())):
            return np.asarray(np.array([args])).astype('float32')

        featureList = [0] * self.featureVectorSize

        for i in range(self.featureVectorSize):
            featureList[i] = args[self.attributesListOrder[i]]

        return np.asarray(np.array(featureList)).astype('float32')

    def getRandomVectorFromDict(self, args):
        if (type(args) == type(list())):
            return np.asarray(np.array(args)).astype('float32')
        elif (type(args) ==  type(tf.random.normal(shape=[0]))):
            return args
        elif args == None:
            return None

        randomList = [0] * self.randomVectorSize

        try:
            for i in range(self.randomVectorSize):
                randomList[i] = args["rv{}".format(i)]
        except:
            return None

        return np.asarray(np.array(randomList)).astype('float32')

    def getRandomVector(self):
        return tf.random.normal(shape=[1, self.randomVectorSize])

    def setAttributesOrder(self, attributes):
        self.attributesListOrder = attributes

    def getAttributesOrder(self):
        return self.attributesListOrder
       
    def setModelPath(self, path):
        self.modelPath = path

    def setModelVersion(self, version):
        self.modelVersion = version

    def setModelPath(self, path):
        self.modelPath = path

    def loadGenerator(self):
        self.generator = keras.models.load_model(self.modelPath.format(self.modelVersion))

    def saveImage(self, imageName = None):
        data = (self.gImage.numpy() * 255)[0]
        data = data.astype('uint8')

        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        self.outputImagePath = os.path.join(self.outputFolder, imageName or self.outputImage)
        im.save(self.outputImagePath)

    def showImage(self):
        data = (self.gImage.numpy() * 255)[0]
        data = data.astype('uint8')
        plt.imshow(data, cmap=plt.cm.viridis)
        plt.axis('off')
        plt.show()

    def getImage(self, features, randomVector = None, autoSave = False, autoShow = False, imageName = None):
        features = self.getFeatureVectorFromDict(features)
        randomNoise = self.getRandomVectorFromDict(randomVector) 
        if randomNoise == None:
            randomNoise = self.getRandomVector()

        self.gImage = self.generator([features, randomNoise], training=False)
        
        if autoSave:
            if imageName[-4:] != ".png" and imageName[-4:] != ".jpg" and imageName[-5:] != ".jpeg":
                imageName += ".png"
            self.saveImage(imageName)

        if autoShow:
            self.showImage()

        return self.gImage

    def getOutputImagePath():
        return self.outputImagePath

if __name__ == "__main__":
    print(tf.random.normal(shape=[1]), type(tf.random.normal(shape=[0])))
    gen = generatorModule()
    attributesOrder = gen.getAttributesOrder()
    attributes = gen.getDummyAttributes()
    randomNoise = gen.getRandomVector()

    maxLen = 20
    print("-"*100)
    print("Attributes passed")
    print("-"*100)
    for i in range(gen.getAttributesSize()):
        print(attributesOrder[i], " " * (maxLen - len(attributesOrder[i])), gen.getDummyAttributes()[i])
    print("-"*100)
    print("Random Latent: ")
    for i in np.array(randomNoise[0]):
        print(i, end="  ")
    print()
    print("-"*100)

    gen.getImage(attributes, randomVector = randomNoise, autoShow = True, autoSave = True, imageName = "test")
    gen.getImage(attributes, autoShow = True, autoSave = True, imageName = "test")
    gen.getImage(attributes, autoShow = True, autoSave = True, imageName = "test")
    gen.getImage(attributes, autoShow = True, autoSave = True, imageName = "test")