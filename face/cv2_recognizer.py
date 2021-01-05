import cv2
import face
from face.recognizer import Recognizer


class CV2Recognizer(Recognizer):
    def extract_faces(self, image):
        array = face.image2array(image)
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        faces = []
        for (x, y, w, h) in detected:
            faces.append(face.array2image(array[y: y + h, x: x + w]))
        return faces
