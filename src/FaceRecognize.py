import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import json
import sys

class FaceIdentifier():
    def __init__(self):
        self.basePath = "Dummy Dataset"
        self.imagesPath = os.path.join(self.basePath, "imgs")
        self.detailsPath = os.path.join(self.basePath, "details")
        self.testPath = os.path.join(self.basePath, "to test")
        self.tolerance = 0.6
        self.model = "cnn" # "hog" or "cnn"

        self.known_faces = []
        self.known_names = []

    def loadFaces(self):
        for name in os.listdir(self.imagesPath):
            for fileName in os.listdir(os.path.join(self.imagesPath, name)):
                image = face_recognition.load_image_file(os.path.join(self.imagesPath, name, fileName))
                encoding = face_recognition.face_encodings(image)[0]

                self.known_faces.append(encoding)
                self.known_names.append(name)

    def getFaceID(self, imagePath = None, image = None):
        if imagePath:
            image = cv2.imread(imagePath, 1)

        location = face_recognition.face_locations(image, model=self.model)
        if len(location) == 0:
            return "none"

        encoding = face_recognition.face_encodings(image, location)[0]
        location = location[0]

        IDs = []

        results = face_recognition.compare_faces(self.known_faces, encoding, self.tolerance)

        if True in results:
            for index in range(len(results)):
                if results[index]:
                    IDs.append(self.known_names[index])

            return IDs
        else:
            return ["none"]
                
    def getDetails(self, _id):      
        jsonFileName = "{}.json"
        jsonFilePath = os.path.join(self.detailsPath, jsonFileName).format(_id)

        with open(jsonFilePath, "r") as f:
            jsonObj = json.load(f)

        return jsonObj

if __name__ == "__main__":
    fi = FaceIdentifier()
    fi.loadFaces()

    if len(sys.argv) > 1:
        paths = sys.argv[1:]

        for path in paths:
            print(fi.getFaceID(path)[0])
    else:
        temp = "Dummy Dataset\\to test\\dummy-test{}.png"
        print("Loaded")

        for i in range(9):
            print(fi.getFaceID(temp.format(i)))
            print("--")

        input("Enter something")

        print(fi.getDetails("1009"))
