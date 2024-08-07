import unittest
from src.face_detection.detect_faces import detect_faces

class TestDetectFaces(unittest.TestCase):
    def test_detect_faces(self):
        image_path = 'test_data/sample.jpg'
        faces = detect_faces(image_path)
        self.assertGreaterEqual(len(faces), 1)

if __name__ == '__main__':
    unittest.main()
