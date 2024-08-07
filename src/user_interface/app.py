import os
import certifi
import ssl
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_embedding(model, face_pixels):
    face_pixels = tf.convert_to_tensor(face_pixels, dtype=tf.float32)
    face_pixels = tf.image.resize(face_pixels, (224, 224))  # Resize to 224x224 for MobileNetV2
    face_pixels = tf.expand_dims(face_pixels, axis=0)
    face_pixels = face_pixels / 255.0  # Normalize the image
    embeddings = model(face_pixels)
    return embeddings.numpy().flatten()

def load_embeddings(embeddings_dir):
    if not os.path.exists(embeddings_dir):
        print("Embeddings directory does not exist")
        return {}
    
    embeddings = {}
    files = os.listdir(embeddings_dir)
    print(f"Files in embeddings directory: {files}")
    
    for filename in files:
        if filename.endswith('.npy'):
            identity = filename.split('.')[0]
            try:
                embeddings[identity] = np.load(os.path.join(embeddings_dir, filename), allow_pickle=True)
                print(f"Loaded embedding for {identity} with shape {embeddings[identity].shape}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    print(f"Loaded embeddings: {embeddings}")
    return embeddings

def recognize_face(face_embedding, embeddings, threshold=0.5):
    min_dist = float('inf')
    identity = None
    for name, db_embedding in embeddings.items():
        print(f"Comparing with {name}, db_embedding shape: {db_embedding.shape}")
        dist = np.linalg.norm(face_embedding - db_embedding)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            identity = name
    return identity, min_dist

@app.route('/')
def index():
    return "Welcome to the Face Recognition API. Use the /recognize endpoint to upload an image for face recognition."

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        image = Image.open(file)
        image = np.array(image)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return jsonify({'error': 'No faces detected'})
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        face_embedding = get_embedding(model, face)
        print(f"Face embedding shape: {face_embedding.shape}")
        embeddings = load_embeddings('embeddings')
        if not embeddings:
            return jsonify({'error': 'No embeddings found'})
        identity, min_dist = recognize_face(face_embedding, embeddings)
        if identity is None:
            return jsonify({'error': 'Face not recognized'})
        return jsonify({'identity': identity, 'distance': float(min_dist)})
    return jsonify({'error': 'File not allowed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
