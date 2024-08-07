import cv2
import os
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return image

def preprocess_images(image_dir, processed_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        processed_image = preprocess_image(image_path)
        np.save(os.path.join(processed_dir, filename.split('.')[0] + '.npy'), processed_image)
