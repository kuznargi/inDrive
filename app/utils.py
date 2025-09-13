from deepface import DeepFace
import cv2
import numpy as np

def get_face_embedding(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    result = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
    if result:
        return np.array(result[0]["embedding"], dtype=np.float32).tobytes()
    return None

