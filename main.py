from LatestVideoCapture import LatestVideoCapture
from cat import Cat
from counter import Counter
from deterrent import Deterrent
import json

import cv2

from ultralytics import YOLO

min_confidence = 0.2
# counter = Counter("counter.json")
deterrent = Deterrent(5, ["Get off the counter!", "What are you doing! Get off!", "No, no, no!", "Get down!", "Off! Now!"])
model = YOLO("yolov8n.pt")

cap = LatestVideoCapture("rocky_ginger.mp4")
#cap = cv2.VideoCapture(4)


def set_counters(frame, win_name: str):
    print("Left click to select n-gon corners")
    print("Middle click to add current n-gon (needs at least 3 corners) and start a new one")
    print("Hit 's' when done")
    counters = {"counters": []}

    current_ngon = []

    def mouse_cb(event, x, y, flags, param):
        for v in counters["counters"]:
            coordinates = v["coordinates"]
            for i, xy in enumerate(coordinates):
                start = (xy[0], xy[1])

                # Close the ngone
                if i == len(coordinates) - 1:
                    end_idx = 0
                else:
                    end_idx = i + 1

                end = (coordinates[end_idx][0], coordinates[end_idx][1])

                cv2.line(frame, start, end, (255, 0, 0))

        for i, xy in enumerate(current_ngon):
            if i == len(current_ngon) - 1:
                break

            start = (xy[0], xy[1])
            end = (current_ngon[i + 1][0], current_ngon[i + 1][1])

            cv2.line(frame, start, end, (255, 0, 0))

        if event == cv2.EVENT_LBUTTONUP:
            current_ngon.append([x, y])

        if event == cv2.EVENT_MBUTTONUP:
            if len(current_ngon) < 3:
                print("Need to have at least 3 corners to save an ngon")
                return

            ngon = current_ngon.copy()
            counter_id = len(counters["counters"]) + 1
            counters["counters"].append({"id": counter_id, "name": f"Counter_{counter_id}", "coordinates": ngon})
            current_ngon.clear()

    cv2.setMouseCallback(win_name, mouse_cb)

    while True:
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cv2.setMouseCallback(win_name, lambda *args: None)

    return counters


cv2.namedWindow("Cat Detector", cv2.WINDOW_AUTOSIZE)

if cap.isOpened():
    s, f = cap.read()
    if s:
        counters = set_counters(f, "Cat Detector")
        with open("set_counters.json", "w") as f:
            json.dump(counters, f, indent=4)

counter = Counter("set_counters.json")

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
