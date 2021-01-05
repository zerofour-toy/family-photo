from google.cloud import vision
import face


class GCPRecognizer(face.Recognizer):
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def extract_faces(self, image):
        vimage = vision.Image(content=face.image2bytes(image))
        response = self.client.face_detection(image=vimage)
        annotations = response.face_annotations
        faces = []
        for annotation in annotations:
            left_top = annotation.bounding_poly.vertices[0]
            right_bottom = annotation.bounding_poly.vertices[2]
            faces.append(image.crop((left_top.x, left_top.y, right_bottom.x, right_bottom.y)))
        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))
        return faces
