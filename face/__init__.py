import os
import shutil
import numpy
from PIL import Image
from .fr_recognizer import FRRecognizer
from .cv2_recognizer import CV2Recognizer


def image2array(image):
    return numpy.array(image)


def array2image(array):
    return Image.fromarray(array)


def extract_faces(image_dir, face_dir, recognizer):
    if recognizer == "FR":
        recognizer = FRRecognizer()
    elif recognizer == "CV2":
        recognizer = CV2Recognizer()
    else:
        raise NotImplementedError("Unknown recognizer")

    dir_files = os.listdir(image_dir)
    shutil.rmtree(face_dir, ignore_errors=True)
    os.mkdir(face_dir)
    for file in dir_files:
        fname, ext = os.path.splitext(file)
        if ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            fpath = "./images/" + file
            image = Image.open(fpath)
            faces = recognizer.extract_faces(image)
            for i in range(len(faces)):
                face = faces[i]
                wpath = face_dir + '/' + fname + '-' + str(i) + '.jpg'
                print('Write ' + wpath)
                face.save(wpath)
