from LatestVideoCapture import LatestVideoCapture
from cat import Cat
from counter import Counter
from deterrent import Deterrent

import cv2

from ultralytics import YOLO

min_confidence = 0.2
counter = Counter("counter.json")

# "southpark" has an md5hash of 6e77e6ec9ed4394409126116a070253a,
# which has been replaced with a southpark soundbite mp3 file
deterrent = Deterrent(5, ["southpark", "Get off the counter!", "What are you doing! Get off!", "No, no, no!", "Get down!", "Off! Now!"])

model = YOLO("yolov8x.pt")

cap = LatestVideoCapture("output_left.mp4")

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

        annotated_frame = frame.copy()

        counter_colour = (255, 0, 255)
        for cat in cats:
            on_counter = cat.on_counter(counter.to_list())
            if on_counter:
                deterrent.try_deter(cat.centre)
                counter_colour = (0, 0, 255)
            annotated_frame = cat.draw(annotated_frame, on_counter)
        annotated_frame = counter.draw(annotated_frame, counter_colour)

        cv2.imshow("Cat Detector", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
