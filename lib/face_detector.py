import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNet('../face_detector/deploy.protoxt',
                                   '../face_detector/res10_300x300_ssd_iter_140000.caffemodel')

    def get_detections(self, image, threshold:float = .6, margin: int = 3) -> list:
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        height, width = image.shape[:2]
        contours = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                # Get more that just the face
                start_x -= margin
                start_y -= margin
                end_x += margin
                end_y += margin

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (start_x, start_y) = (max(0, start_x), max(0, start_y))
                (end_x, end_y) = (min(width - 1, end_x), min(height - 1, end_y))

                contours.append((start_x, start_y, end_x, end_y))
        return contours
