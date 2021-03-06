import face_recognition as fr
import face
from face.recognizer import Recognizer


class FRRecognizer(Recognizer):
    def extract_faces(self, image):
        array = face.image2array(image)
        faces = []
        locations = fr.face_locations(array)
        for location in locations:
            top, right, bottom, left = location
            faces.append(face.array2image(array[top:bottom, left:right]))
        return faces
