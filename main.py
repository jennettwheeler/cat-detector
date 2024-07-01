import cv2

from ultralytics import YOLO


class Cat:
    def __init__(self, confidence, xyxy):
        self.confidence = confidence
        self.start_point = (int(xyxy[0]), int(xyxy[1]))
        self.end_point = (int(xyxy[2]), int(xyxy[3]))

    def __str__(self):
        return f"Cat: {self.start_point} to {self.end_point} Confidence: {self.confidence * 100:.2f}%"

    def draw(self, frame):
        color = (0, 0, 255)
        thickness = 3
        font_scale = 0.6
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(frame, f"Cat (Confidence: {self.confidence * 100:.2f}%)", (self.start_point[0]+5, self.start_point[1]+20), font, font_scale, color, 1, cv2.LINE_AA)
        return cv2.rectangle(image, self.start_point, self.end_point, color, thickness)


min_confidence = 0.2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        cats = []
        for r in results:
            for box in r.boxes:
                class_name = model.names[int(box.cls)]
                if class_name == "cat":
                    cat = Cat(float(box.conf), box.xyxy.numpy()[0])
                    if cat.confidence > min_confidence:
                        cats.append(cat)
                        print(f"{cat}")

        annotated_frame = frame
        for cat in cats:
            annotated_frame = cat.draw(annotated_frame)


        cv2.imshow("Cat Detector", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
