import cv2
import shapely.affinity
from shapely.geometry import Polygon, Point
import numpy as np

from ultralytics import YOLO


class Cat:
    def __init__(self, confidence, xyxy):
        self.confidence = confidence
        self.start_point = (round(xyxy[0]), round(xyxy[1]))
        self.end_point = (round(xyxy[2]), round(xyxy[3]))
        self.centre = (round((xyxy[0] + xyxy[2]) / 2), round((xyxy[1] + xyxy[3]) / 2))
        circle = Point(self.centre).buffer(1)
        ellipse_x_scale = round((xyxy[2] - xyxy[0]) / 2)
        ellipse_y_scale = round((xyxy[3] - xyxy[1]) / 2)
        self.shape = shapely.affinity.scale(circle, ellipse_x_scale, ellipse_y_scale)


    def __str__(self):
        return f"Cat: {self.start_point} to {self.end_point} Confidence: {self.confidence * 100:.2f}%"

    def draw(self, frame, on_counter):
        color = (0, 0, 255) if on_counter else (0, 255, 0)
        font_scale = 0.6
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"Cat (Confidence: {self.confidence * 100:.2f}%)"

        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = round(self.centre[0] - (text_size[0] / 2))
        text_y = round(self.centre[1] + (text_size[1] / 2))
        image = cv2.putText(frame, label, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

        ellipse_thickness = 3
        return cv2.polylines(image, [np.array(self.shape.exterior.coords, dtype=np.int32)], True, color, ellipse_thickness)

    def on_counter(self, counter_polygons):
        for polygon in counter_polygons:
            if self.shape.intersects(polygon):
                return True
        return False


min_confidence = 0.2

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

        counter = Polygon([(50,100), (200,50), (300,70), (450,150), (220,210), (120,200)])
        counter_colour = (255, 0, 255)
        for cat in cats:
            on_counter = cat.on_counter([counter])
            if on_counter:
                counter_colour = (0, 0, 255)
            annotated_frame = cat.draw(annotated_frame, on_counter)

        counter_img = np.zeros_like(frame, np.uint8)
        counter_img = cv2.fillPoly(counter_img, [np.array(counter.exterior.coords, dtype=np.int32)], counter_colour)
        counter_mask = counter_img.astype(bool)
        annotated_frame[counter_mask] = cv2.addWeighted(frame, 0.2, counter_img, 0.8, 0)[counter_mask]

        cv2.imshow("Cat Detector", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
