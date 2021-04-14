import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import json

class FaceIdentifier():
    def __init__(self):
        self.basePath = "Dummy Dataset"
        self.imagesPath = os.path.join(self.basePath, "imgs")
        self.detailsPath = os.path.join(self.basePath, "details")
        self.testPath = os.path.join(self.basePath, "to test")

        self.encoded = {}

        for dirpath, dnames, fnames in os.walk(self.imagesPath):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file(os.path.join(self.imagesPath, f))
                    encoding = fr.face_encodings(face)[0]
                    self.encoded[f.split(".")[0]] = encoding
        
        self.faces_encoded = list(self.encoded.values())
        self.known_face_names = list(self.encoded.keys())

    def getFaceID(self, imagePath = None, image = None):
        if imagePath:
            image = cv2.imread(imagePath, 1)

        face_locations = face_recognition.face_locations(image)
        unknown_face_encodings = face_recognition.face_encodings(image, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.faces_encoded, face_encoding)
            name = "none"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)
        
        return face_names

    def getDetails(self, _id):      
        jsonFileName = "{}.json"
        jsonFilePath = os.path.join(self.detailsPath, jsonFileName).format(_id)

        with open(jsonFilePath, "r") as f:
            jsonObj = json.load(f)

        return jsonObj

if __name__ == "__main__":
    fi = FaceIdentifier()

    temp = "Dummy Dataset\\to test\\test{}.png"
    print("Loaded")

    for i in range(9):
        print(fi.getFaceID(temp.format(i)))
        print("--")

    input("Enter something")

    print(fi.getDetails("1009"))
