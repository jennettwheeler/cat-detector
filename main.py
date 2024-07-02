from cat import Cat
from counter import Counter

import cv2
from shapely.geometry import Polygon
import numpy as np

from ultralytics import YOLO

min_confidence = 0.2
counter = Counter("counter.json")

model = YOLO("yolov8n.pt")

# cap = cv2.VideoCapture("filepath.mp4")
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        cats = []
        for r in results:
            for box in r.boxes:
                class_name = model.names[int(box.cls)]
                if class_name == "person":  # Easier to test in office
                    cat = Cat(float(box.conf), box.xyxy.numpy()[0])
                    if cat.confidence > min_confidence:
                        cats.append(cat)
                        print(f"{cat}")

        annotated_frame = frame.copy()

        counter_colour = (255, 0, 255)
        for cat in cats:
            on_counter = cat.on_counter(counter.to_list())
            if on_counter:
                counter_colour = (0, 0, 255)
            annotated_frame = cat.draw(annotated_frame, on_counter)
        annotated_frame = counter.draw(annotated_frame, counter_colour)

        cv2.imshow("Cat Detector", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
