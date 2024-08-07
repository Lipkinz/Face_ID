import os
import certifi
import ssl
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
import urllib.request

def configure_ssl():
    context = ssl.create_default_context(cafile=certifi.where())
    https_handler = urllib.request.HTTPSHandler(context=context)
    opener = urllib.request.build_opener(https_handler)
    urllib.request.install_opener(opener)

configure_ssl()

def load_model_with_context():
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    model = hub.load(model_url)
    return model

model = load_model_with_context()

images_dir = "/Users/tomerkanelstein/Desktop/Vertical/face_id_project/images"
embeddings_dir = "/Users/tomerkanelstein/Desktop/Vertical/face_id_project/embeddings"

if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

def process_image(image_path, identity):
    print(f"Processing {image_path}")
    image = Image.open(image_path)
    image = np.array(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print(f"No faces detected in {image_path}")
        return
    (x, y, w, h) = faces[0]
    face = image[y:y + h, x:x + w]
    face_embedding = get_embedding(model, face)
    print(f"Embedding shape for {identity}: {face_embedding.shape}")
    np.save(os.path.join(embeddings_dir, f"{identity}.npy"), face_embedding)
    print(f"Embedding for {identity} saved to {os.path.join(embeddings_dir, f'{identity}.npy')}")

def get_embedding(model, face_pixels):
    face_pixels = tf.convert_to_tensor(face_pixels, dtype=tf.float32)
    face_pixels = tf.image.resize(face_pixels, (224, 224))  # Resize to 224x224 for MobileNetV2
    face_pixels = tf.expand_dims(face_pixels, axis=0)
    face_pixels = face_pixels / 255.0  # Normalize the image
    embeddings = model(face_pixels)
    return embeddings.numpy().flatten()

# Clear existing embeddings
for filename in os.listdir(embeddings_dir):
    if filename.endswith('.npy'):
        os.remove(os.path.join(embeddings_dir, filename))

# Generate new embeddings
for filename in os.listdir(images_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        identity = os.path.splitext(filename)[0]
        image_path = os.path.join(images_dir, filename)
        process_image(image_path, identity)
