import os
import certifi
import ssl
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Create an SSL context using certifi's CA bundle
context = ssl.create_default_context(cafile=certifi.where())

def load_model_with_context():
    # URL to the model
    model_url = "https://tfhub.dev/google/facenet/1"

    # Load the model using the SSL context
    model = hub.load(model_url, tags=[], options=tf.saved_model.LoadOptions(
        experimental_io_device='/job:localhost'
    ))
    return model

model = load_model_with_context()

def get_embedding(model, face_pixels):
    face_pixels = tf.convert_to_tensor(face_pixels, dtype=tf.float32)
    face_pixels = tf.image.resize(face_pixels, (160, 160))
    face_pixels = (face_pixels - 127.5) / 128.0  # Normalize the image
    embeddings = model(face_pixels)
    return embeddings.numpy()

def load_embeddings(embeddings_dir):
    embeddings = {}
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.npy'):
            identity = filename.split('.')[0]
            embeddings[identity] = np.load(os.path.join(embeddings_dir, filename))
    return embeddings

def recognize_face(face_embedding, embeddings, threshold=0.5):
    min_dist = float('inf')
    identity = None
    for name, db_embedding in embeddings.items():
        dist = np.linalg.norm(face_embedding - db_embedding)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            identity = name
    return identity, min_dist
