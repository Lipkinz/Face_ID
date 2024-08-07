import unittest
from src.data_preprocessing.preprocess import preprocess_image

class TestPreprocess(unittest.TestCase):
    def test_preprocess_image(self):
        image_path = 'test_data/sample.jpg'
        processed_image = preprocess_image(image_path)
        self.assertEqual(processed_image.shape, (160, 160, 3))

if __name__ == '__main__':
    unittest.main()
