#!/usr/bin/env python
# coding: utf-8

# In[59]:


from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from PIL import Image
import numpy as np
from src.gan import getGanModel

import cv2
import pandas as pd
import os
from tqdm import tqdm


# In[60]:


def plotSaveImage(image, savePath = ''):
    data = (gImage.numpy() * 255)[0]
    plt.imshow(data)
    plt.show()
    if len(savePath):
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(savePath)


# In[61]:


gan, generator, discriminator = getGanModel()


# In[62]:


# Test Generator image
features = tf.random.normal(shape=[1, 40])
randomNoise = tf.random.normal(shape=[1, 100])
gImage = generator([features, randomNoise], training=False)
# gImage = generator([randomNoise, features], training=False)
plotSaveImage(gImage)


# In[63]:


decision = discriminator([features, gImage])
print(decision)


# In[74]:


picsPath = 'P:/GAN Learning/Face_Generation/datasets/29561_37705_bundle_archive/img_align_celeba/processed/'
csvPath = 'P:/GAN Learning/Face_Generation/datasets/29561_37705_bundle_archive/list_attr_celeba.csv'
data = pd.read_csv(csvPath)
numpyData = data.values
# print(len(numpyData))


# In[78]:


batchSize = 25
start = 0
iterations = 10
randomNoiseLength = 100


picNames = numpyData[start:start+batchSize, 0]
attributes = numpyData[start:start+batchSize, 1:]
attributes = np.asarray(attributes).astype('float32')
attributes += 0.05 * np.random.random(attributes.shape)


# In[79]:


def getMetaData(rawData, start, batchSize, rootPath = ''):
    picNames = rawData[start:start+batchSize, 0]
    attributes = rawData[start:start+batchSize, 1:]
    attributes = np.asarray(attributes).astype('float32')
    attributes += 0.05 * np.random.random(attributes.shape)
    
    if len(rootPath):
        images = getImages(rootPath, picNames)
        return (images, attributes)
    return (picNames, attributes)
    


# In[80]:


print(attributes.shape, type(attributes))
print(picNames.shape)
randomVector = np.random.normal(size=(batchSize, randomNoiseLength))
print(randomVector.shape, type(randomVector))
print(features.shape)
print(randomNoise.shape)


# In[81]:


print(attributes)


# In[82]:


def getImages(rootPath, imageList):
    images = []
    for i in imageList:
        temp = cv2.imread(rootPath + i)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp = temp / 255
        images.append(temp)
    return np.array(images)


# In[83]:


images = getImages(picsPath, picNames)


# In[84]:


print(images.shape)


# In[85]:


plt.figure(1, figsize=(10, 10))
for i in range(batchSize):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.axis('off')
plt.show()
    


# In[98]:


discriminatorLosses = []
adversaryLosses = []
start = 42500 + 25000
iterations = 1000

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
    
    start += batchSize
    if start > len(numpyData) - batchSize:
        start = 0
        
    


# In[99]:


tempGeneratedImages = generatedImages * 255

plt.figure(1, figsize=(10, 10))
for i in range(batchSize):
    plt.subplot(5, 5, i+1)
    plt.imshow(tempGeneratedImages[i])
    plt.axis('off')
plt.show()


# In[ ]:




