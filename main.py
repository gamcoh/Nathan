import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from lib.face_detector import FaceDetector

WIDTH, HEIGHT = 200, 200

cap = cv2.VideoCapture(0)
cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, HEIGHT)

labels = [
    '0_2',
    '14_20',
    '21_32',
    '33_43',
    '3_6',
    '44_53',
    '54_100',
    '7_13'
]

model = load_model('./models/mobilenet_128_img_200_b64.h5')
face_detector = FaceDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = img_to_array(frame)
    frame /= 255
    frame = np.expand_dims(frame, axis=0)

    ages_in_image = []
    faces = face_detector.get_detections(frame)
    for face in faces:
        sx, sy, ex, ey = face
        mask = frame[sy:ey, sx:ex]
        prediction = model.predict([mask])
        prediction = list(prediction[0])
        index = prediction.index(max(prediction))
        label = labels[index]
        ages_in_image.append(label)

    original_frame = cv2.flip(original_frame, 1)
    cv2.imshow('osef', original_frame)
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
