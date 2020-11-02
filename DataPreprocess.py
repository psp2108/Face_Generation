import pandas as pd
import shutil
import os
import json
import glob
from tqdm import tqdm
from PIL import Image

with open("config.json", "r") as f:
    dataset = json.load(f)['DatasetDetails']

def resizeImagesTo128x128():
    picsDirectory = os.path.join(dataset['DatasetRootPath'], dataset['DatasetImages']).replace("/", "\\")
    picsProcessedDirectory = os.path.join(dataset['DatasetRootPath'], dataset['DatasetProcessedImages']).replace("/", "\\")

    if not os.path.exists(picsProcessedDirectory):
        os.makedirs(picsProcessedDirectory)

        originalWidth = 178
        originalHeight = 218
        difference = (originalHeight - originalWidth) // 2

        newWidth = 128
        newHeight = 128

        cropRect = (0, difference, originalWidth, originalHeight - difference)

        for eachImage in tqdm(os.listdir(picsDirectory)):
            image = Image.open(os.path.join(picsDirectory, eachImage)).crop(cropRect)
            image.thumbnail((newWidth, newHeight), Image.ANTIALIAS)
            image.save(os.path.join(picsProcessedDirectory, eachImage))


# 1. Combine Excels to single CSV
def combineExcels(): # Mandatory
    os.chdir("E:/sem 6/project/datasets/")

    path1 = "E:/sem 6/project/datasets/list_attr_celeba.csv"
    path2 = "E:/sem 6/project/datasets/list_bbox_celeba.csv"
    path3 = "E:/sem 6/project/datasets/list_landmarks_align_celeba.csv"

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)

    df = df1.merge(df2, how = 'outer', on = 'image_id')
    df.to_csv("test1.csv", index = False)

    path4 = "E:/sem 6/project/datasets/test1.csv"
    df4 = pd.read_csv(path4)
    df = df4.merge(df3, how = 'outer', on = 'image_id')
    df.to_csv("test2.csv", index = False)

# 2. Remove blurred images
def deleteBlurredImages():
    pass

# 3. Manual Filtering 
def selectiveDelete():
    pass


resizeImagesTo128x128()
combineExcels()
# deleteBlurredImages()
selectiveDelete()