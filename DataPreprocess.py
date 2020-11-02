import pandas as pd
import os
import json
from tqdm import tqdm
from PIL import Image

with open("config.json", "r") as f:
    jsonFile = json.load(f)
    CSVDetails = jsonFile['CSVDetails']
    dataset = jsonFile['ImageDetails']
    

def resizeImagesTo128x128():
    picsDirectory = os.path.join(dataset['ImageRootPath'], dataset['ImageImages']).replace("/", "\\")
    picsProcessedDirectory = os.path.join(dataset['ImageRootPath'], dataset['ImageProcessedImages']).replace("/", "\\")

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
    CSVRootPath = CSVDetails['CSVRootPath']
    CSVList = CSVDetails['CSVList']
    CombinedCSV = CSVDetails['CombinedCSV']
    if len(CSVList) == 0:
        return
    df = pd.read_csv(os.path.join(CSVRootPath, CSVList[0]).replace("/", "\\"))
    for i in range(1,len(CSVList)):
        CSVFile = pd.read_csv(os.path.join(CSVRootPath, CSVList[i]).replace("/", "\\"))
        df = df.merge(CSVFile, how = 'outer', on = 'image_id')
    df.to_csv(os.path.join(CSVRootPath, CombinedCSV).replace("/", "\\"), index = False)




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