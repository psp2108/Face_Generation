import cv2
from tqdm import tqdm
import os
import json

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getFaceCount(facePath):
    img = cv2.imread(facePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

with open("config.json", "r") as f:
    jsonFile = json.load(f)
    imageDetails = jsonFile['ImageDetails']

imageDataset = os.path.join(imageDetails['ImageRootPath'], imageDetails['ImageProcessedImages'], '{}.jpg')
# imageDataset = "datasets\\29561_37705_bundle_archive\\img_align_celeba\\processed\\{}.jpg"

def fillZeros(num, noOfZeros = 6):
    temp = str(num)
    temp = ('0'*(noOfZeros - len(temp))) + temp
    return temp

print(fillZeros(1))

defectiveList = []

for i in tqdm(range(1, 202600)):
# for i in tqdm(range(1, 100)):
    faces = getFaceCount(imageDataset.format(fillZeros(i)))
    if (len(faces)) != 1:
        defectiveList.append((fillZeros(i), len(faces), ))
#         print(fillZeros(i),len(faces))

print(len(defectiveList))

file = open("ListOfImproperImages.txt", "a")
for i in defectiveList:
    file.write("{}.jpg\n".format(i[0]))
file.close()

from  matplotlib import pyplot as plt
count = 0
for i in defectiveList:
    if i[1] == 0:
        count += 1
#         img = cv2.imread(imageDataset.format(i[0]),1)
#         img2 = img[:,:,::-1] 
#         plt.imshow(img2)
#         plt.show()
        print(i)
        
print(count)