# Face ID Project

## Overview
This project implements a Face ID system using deep learning techniques. It includes face detection, face recognition, and a web-based user interface for enrolling and recognizing faces.

## Features
- Face detection using MTCNN
- Face recognition using a deep learning model (FaceNet)
- Real-time face recognition using OpenCV
- Web-based user interface for enrolling new faces and recognizing faces

## File Structure
├── data/
│ ├── raw/
│ ├── processed/
│ └── models/
├── src/
│ ├── data_preprocessing/
│ │ ├── preprocess.py
│ │ └── init.py
│ ├── face_detection/
│ │ ├── detect_faces.py
│ │ └── init.py
│ ├── face_recognition/
│ │ ├── recognize_faces.py
│ │ └── init.py
│ ├── real_time_processing/
│ │ ├── real_time_recognition.py
│ │ └── init.py
│ ├── user_interface/
│ │ ├── app.py
│ │ ├── templates/
│ │ │ ├── base.html
│ │ │ ├── index.html
│ │ │ └── enroll.html
│ │ └── static/
│ └── init.py
├── notebooks/
│ ├── data_exploration.ipynb
│ ├── model_training.ipynb
│ └── evaluation.ipynb
├── tests/
│ ├── test_preprocess.py
│ ├── test_detect_faces.py
│ ├── test_recognize_faces.py
│ ├── test_real_time_recognition.py
│ └── init.py
├── requirements.txt
├── README.md
└── .gitignore
