import cv2
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from lib.face_detector import FaceDetector

WIDTH, HEIGHT = 300, 300

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

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

AGES_NOT_ALLOWED = [0, 1, 5, 8]

model = load_model('./models/mobilenet_128_img_200_b64.h5')
face_detector = FaceDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ages_in_image = ''
    faces = face_detector.get_detections(frame, margin=4)
    if len(faces) < 1:
        continue

    for face in faces:
        sx, sy, ex, ey = face
        cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 0, 0), thickness=2)
        mask = frame[sy:ey, sx:ex]
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (200, 200))
        mask = img_to_array(frame)
        mask /= 255
        mask = np.expand_dims(frame, axis=0)
        prediction = model.predict([mask])
        prediction = list(prediction[0])
        index = prediction.index(max(prediction))
        label = labels[index]
        ages_in_image = label

    print(ages_in_image)

    # Lock the screen if a child face were found
    index = labels.index(ages_in_image)
    if index in AGES_NOT_ALLOWED:
        os.system('systemctl suspend')
        break

    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
