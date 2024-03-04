import cv2
import os
import numpy as np
from PIL import Image

path = 'samples'

# Check if cv2.face submodule is available
if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
else:
    recognizer = cv2.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def Images_And_Labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        gray_img = Image.open(imagePath).convert('L')
        img_arr = np.array(gray_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_arr)

        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

print("Training faces. It will take a few seconds. Wait ...")

faces, ids = Images_And_Labels(path)
recognizer.train(faces, np.array(ids))

recognizer.save('trainer/trainer.yml')  # Save the trained model

print("Model trained. Now we can recognize your face.")
