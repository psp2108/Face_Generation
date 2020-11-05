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

def plotSaveImage(image, savePath = ''):
    data = (image.numpy() * 255)[0]
    plt.imshow(data)
    plt.show()
    if len(savePath):
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(savePath)

def getImages(rootPath, imageList):
    images = []
    for i in imageList:
        temp = cv2.imread(rootPath + i)
        # https://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp = temp / 255
        images.append(temp)
    return np.array(images)

def getMetaData(rawData, start, batchSize, rootPath = ''):
    picNames = rawData[start:start+batchSize, 0]
    attributes = rawData[start:start+batchSize, 1:]
    attributes = np.asarray(attributes).astype('float32')
    attributes += 0.05 * np.random.random(attributes.shape)
    
    if len(rootPath):
        images = getImages(rootPath, picNames)
        return (images, attributes)
    return (picNames, attributes)

gan, generator, discriminator = getGanModel()

# Test Generator image
randomNoiseLength = 100
features = tf.random.normal(shape=[1, 40])
randomNoise = tf.random.normal(shape=[1, randomNoiseLength])
gImage = generator([features, randomNoise], training=False)
plotSaveImage(gImage)

decision = discriminator([features, gImage])
print(decision)

picsPath = 'P:/GAN Learning/Face_Generation/datasets/29561_37705_bundle_archive/img_align_celeba/processed/'
csvPath = 'P:/GAN Learning/Face_Generation/datasets/29561_37705_bundle_archive/list_attr_celeba.csv'
modelLog = 'P:/GAN Learning/Face_Generation/src/Model Log.csv'

modelLogFile = open(modelLog, "w")
modelLogFile.writelines("Iterations, Discriminator Loss, Adversary Loss, Image number, Loop\n")
modelLogFile.close()

data = pd.read_csv(csvPath)
numpyData = data.values

batchSize = 16
start = 0
iterations = 50000
loop = 0
# stepLimit = len(numpyData) - batchSize
stepLimit = 30000

saveModelInterval = 250
showImageInterval = 10

controlSizeOfSampleImages = 6
_, sampleImagesAttributes = getMetaData(numpyData, 0, controlSizeOfSampleImages**2)
sampleRandomNoise = np.random.normal(size=(controlSizeOfSampleImages**2, randomNoiseLength))

discriminatorLosses = []
adversaryLosses = []

for step in tqdm(range(iterations)):    
    # Fetching the images and their attributes from Hard drive
    realImages, attributes = getMetaData(numpyData, start, batchSize, picsPath)
    
    # Random vector (2nd Generator Input)
    randomVector = np.random.normal(size=(batchSize, randomNoiseLength))
    
    # Generating fake images
    generatedImages = generator.predict([attributes, randomVector])
    
    # Combining 50% fake and 50% real images as well as attributes for discriminator
    combinedImages = np.concatenate([generatedImages, realImages])
    combinedAttributes = np.concatenate([attributes, attributes])
    # 0 => Label for fake images, 1 => Label for real images >>> Used to train Discriminator
    labels = np.concatenate([np.zeros((batchSize, 1)), np.ones((batchSize, 1))])
    labels += .05 * np.random.random(labels.shape)
    
    # Training Discriminator
    discriminatorLoss = discriminator.train_on_batch([combinedAttributes, combinedImages], labels)
    discriminatorLosses.append(discriminatorLoss)
    
    # Preparing to train Generator
    misleadingTargets = np.ones((batchSize, 1))
    misleadingTargets += .05 * np.random.random(misleadingTargets.shape)
    
#     # Random vector (2nd Generator Input)
#     randomVector = np.random.normal(size=(batchSize, randomNoiseLength))
    
    # Training Generator
    adversaryLoss = gan.train_on_batch([randomVector, attributes], misleadingTargets)
    adversaryLosses.append(adversaryLoss)
        
    if step % saveModelInterval == saveModelInterval - 1:
        gan.save_weights('models/gan'+ str(step + 1) +'.h5')
        generator.save_weights('models/generator'+ str(step + 1) +'.h5')
        discriminator.save_weights('models/discriminator'+ str(step + 1) +'.h5')

    if step % showImageInterval == showImageInterval - 1:
        log = 'Iterations: %d/%d, d_loss: %.4f,  a_loss: %.4f. ' % (step + 1, iterations, discriminatorLoss, adversaryLoss)
        print(log)

        modelLogFile = open(modelLog, "a")
        modelLogFile.writelines("%d, %f, %f, %d, %d\n" % (step + 1, discriminatorLoss, adversaryLoss, start, loop))
        modelLogFile.close()

        sampleGeneratedImages = generator.predict([sampleImagesAttributes, sampleRandomNoise])
        control_image = np.ones((128 * controlSizeOfSampleImages, 128 * controlSizeOfSampleImages, 3))
        for i in range(controlSizeOfSampleImages**2):
            x_off = i % controlSizeOfSampleImages
            y_off = i // controlSizeOfSampleImages
            control_image[x_off * 128:(x_off + 1) * 128, y_off * 128:(y_off + 1) * 128, :] = sampleGeneratedImages[i, :, :, :]
        im = Image.fromarray(np.uint8(control_image * 255))
        im.save("myop/%d.png" % (step + 1))
        im.save("myop/latest.tif")

    start += batchSize
    if start > stepLimit:
        loop += 1
        start = 0

tempGeneratedImages = generatedImages * 255

plt.figure(1, figsize=(10, 10))
for i in range(batchSize):
    plt.subplot(5, 5, i+1)
    plt.imshow(tempGeneratedImages[i])
    plt.axis('off')
plt.show()
