import os
import io
import shutil
import numpy
from PIL import Image, ExifTags
from .fr_recognizer import FRRecognizer
from .cv2_recognizer import CV2Recognizer
from .gcp_recognizer import GCPRecognizer


class Recognizer:
    def extract_faces(self, image):
        raise NotImplementedError


def image2array(image):
    return numpy.array(image)


def array2image(array):
    return Image.fromarray(array)


def image2bytes(image):
    bio = io.BytesIO()
    image.save(bio, format="JPEG")
    return bio.getvalue()


def load_image(file_path):
    image = Image.open(file_path)
    if hasattr(image, '_getexif'):
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        e = image._getexif()
        if e is not None:
            exif = dict(e.items())
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    return image


def extract_faces(image_dir, face_dir, recognizer):
    if recognizer == "FR":
        recognizer = FRRecognizer()
    elif recognizer == "CV2":
        recognizer = CV2Recognizer()
    elif recognizer == "GCP":
        recognizer = GCPRecognizer()
    else:
        raise NotImplementedError("Unknown recognizer")

    dir_files = os.listdir(image_dir)
    shutil.rmtree(face_dir, ignore_errors=True)
    os.mkdir(face_dir)
    for file in dir_files:
        fname, ext = os.path.splitext(file)
        if ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            fpath = "./images/" + file
            image = load_image(fpath)
            faces = recognizer.extract_faces(image)
            for i in range(len(faces)):
                face = faces[i]
                wpath = face_dir + '/' + fname + '-' + str(i) + '.jpg'
                print('Write ' + wpath)
                face.save(wpath)

