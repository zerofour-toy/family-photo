import face


if __name__ == "__main__":
	face.extract_faces("./images", "./images/fr-faces", "FR")
	face.extract_faces("./images", "./images/cv2-faces", "CV2")
