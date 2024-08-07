from mtcnn import MTCNN
import cv2

detector = MTCNN()

def detect_faces(image_path):
    image = cv2.imread(image_path)
    faces = detector.detect_faces(image)
    return faces

def draw_faces(image_path, faces):
    image = cv2.imread(image_path)
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return image
