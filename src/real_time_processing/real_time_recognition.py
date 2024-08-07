import cv2
from src.face_detection.detect_faces import detect_faces, draw_faces
from src.face_recognition.recognize_faces import get_embedding, model, load_embeddings, recognize_face

embeddings = load_embeddings('data/processed/')

def real_time_recognition():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        for face in faces:
            x, y, width, height = face['box']
            face_pixels = frame[y:y+height, x:x+width]
            face_embedding = get_embedding(model, face_pixels)
            identity, dist = recognize_face(face_embedding, embeddings)

            if identity:
                label = f"{identity} ({dist:.2f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        cv2.imshow('Real-time Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_recognition()
