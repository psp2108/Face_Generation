import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from matplotlib import pyplot as plt
import tensorflow as tf 

# To resolve GEMM error
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
from PIL import Image
import numpy as np

import json

class GeneratorModule():

    def __init__(self, quick = False):
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

        if not quick:
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
        if (isinstance(args, list)):
            return np.asarray(np.array([args])).astype('float32')

        featureList = [0] * self.featureVectorSize

        for i in range(self.featureVectorSize):
            featureList[i] = args[self.attributesListOrder[i]]

        return np.asarray(np.array([featureList])).astype('float32')

    def getRandomVectorFromDict(self, args):
        if (isinstance(args, list)):
            return np.asarray(np.array([args])).astype('float32')
        elif (type(args) ==  type(tf.random.normal(shape=[0]))):
            return args
        elif (isinstance(args, dict)):
            randomList = [0] * self.randomVectorSize

            try:
                for i in range(self.randomVectorSize):
                    randomList[i] = args["rv{}".format(i)]
            except:
                return None

            return np.asarray(np.array([randomList])).astype('float32')
        else: # args == None:
            return None

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
        print("GENERATOR LOADED")

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
        if str(randomNoise) == 'None':
            randomNoise = self.getRandomVector()

        self.gImage = self.generator([features, randomNoise], training=False)
        
        if autoSave:
            if imageName[-4:] != ".png" and imageName[-4:] != ".jpg" and imageName[-5:] != ".jpeg":
                imageName += ".png"
            self.saveImage(imageName)

        if autoShow:
            self.showImage()

        return self.gImage

    def getOutputImagePath(self):
        return self.outputImagePath

if __name__ == "__main__":
    print(tf.random.normal(shape=[1]), type(tf.random.normal(shape=[0])))
    gen = GeneratorModule()
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

    randomNoise = [1,0,0.1,0.2,0.3,0.4,0.5]

    temp_dict = {
        '5_o_clock_shadow': 0.0,
        'bags_under_eyes': 0.0,
        'big_lips': 0.0,
        'big_nose': 0.0,
        'chubby': 0.0,
        'double_chin': 0.0,
        'goatee': 0.0,
        'heavy_makeup': 0.1,
        'high_cheekbones': 0.1,
        'male': 0.0,
        'mustache': 0.0,
        'narrow_eyes': 0.0,
        'no_beard': 0.1,
        'oval_face': 0.1,
        'pale_skin': 0.0,
        'pointy_nose': 0.1,
        'rosy_cheeks': 0.0,
        'sideburns': 0.1,
        'smiling': 0.1,
        'straight_hair': 0.1,
        'wavy_hair': 0.0,
        'young': 0.1,
        'hair_color': 0.075,
        'hair_size': 0.1,
        'combine_eyebrow': 0.0,
        'rv0': 17.917,
        'rv1': 18.143,
        'rv2': 8.966,
        'rv3': 29.206,
        'rv4': 13.653,
        'rv5': 13.822,
        'rv6': 20.095
    }

    gen.getImage(attributes, randomVector = randomNoise, autoShow = True, autoSave = True, imageName = "test")
    gen.getImage(temp_dict, autoShow = True, autoSave = True, imageName = "test")
    # gen.getImage(attributes, autoShow = True, autoSave = True, imageName = "test")
    # gen.getImage(attributes, autoShow = True, autoSave = True, imageName = "test")