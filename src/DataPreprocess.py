import pandas as pd
import os
import json
from tqdm import tqdm
from PIL import Image

with open("config.json", "r") as f:
    jsonFile = json.load(f)
    CSVDetails = jsonFile['CSVDetails']
    dataset = jsonFile['ImageDetails']
    
# 1. Change the resolution of images
def resizeImagesTo128x128():
    print("Resizing Images ...")
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
        
        print("Resized Images")
    else:
        print("Output folder '"+ picsProcessedDirectory +"' aready exists. Images are either already resized or delete the folder.")
    print("-" * 100)

# 2. Normalize values
def normalizeAttributesFile():
    print("Normalizing Data ...")
    CSVRootPath = CSVDetails['CSVRootPath']
    toNormalize = CSVDetails['NormalizeData']['AttributesFile']
    df = pd.read_csv(os.path.join(CSVRootPath, toNormalize['From']).replace("/", "\\"))

    rawData = df.values
    rawData[:,1:] = (rawData[:,1:]+1)/2
    df = pd.DataFrame(rawData, columns=df.columns)
   
    df.to_csv(os.path.join(CSVRootPath, toNormalize['To']).replace("/", "\\"), index = False)

    print("Files normalized")
    print("-" * 100)

# 3. Combine Excels to single CSV
def combineExcels(): 
    print("Combining CSV files ...")
    CSVRootPath = CSVDetails['CSVRootPath']
    CSVList = CSVDetails['CSVListToCombine']
    CombinedCSV = CSVDetails['CombinedCSV']
    if len(CSVList) == 0:
        print("No files selected to combine")
        return
    df = pd.read_csv(os.path.join(CSVRootPath, CSVList[0]).replace("/", "\\"))
    print("Reading file '"+ CSVList[0] +"'")
    for i in range(1,len(CSVList)):
        print("Reading file '"+ CSVList[i] +"'")
        CSVFile = pd.read_csv(os.path.join(CSVRootPath, CSVList[i]).replace("/", "\\"))
        df = df.merge(CSVFile, how = 'outer', on = 'image_id')
    df.to_csv(os.path.join(CSVRootPath, CombinedCSV).replace("/", "\\"), index = False)

    print("Files combined")
    print("-" * 100)

# 4. Remove blurred images
def deleteBlurredImages():
    pass

# 5. Manual Filtering 
def selectiveDelete():
    pass


resizeImagesTo128x128()
normalizeAttributesFile()
combineExcels()
deleteBlurredImages()
selectiveDelete()