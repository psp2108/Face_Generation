import os
import json
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

resizeImagesTo128x128()