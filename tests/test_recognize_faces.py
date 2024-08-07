import unittest
from src.face_recognition.recognize_faces import get_embedding, recognize_face, model

class TestRecognizeFaces(unittest.TestCase):
    def test_get_embedding(self):
        face_pixels = np.random.rand(160, 160, 3) * 255
        embedding = get_embedding(model, face_pixels)
        self.assertEqual(len(embedding), 128)

    def test_recognize_face(self):
        embeddings = {'test': np.random.rand(128)}
        face_embedding = np.random.rand(128)
        identity, dist = recognize_face(face_embedding, embeddings, threshold=1.0)
        self.assertIsNotNone(identity)

if __name__ == '__main__':
    unittest.main()
