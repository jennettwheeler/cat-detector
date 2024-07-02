import cv2
import json
import numpy as np
from shapely.geometry import Polygon


class Counter:
    counters = []

    def __init__(self, file):
        with open(file) as counter_file:
            data = json.load(counter_file)
            for counter in data["counters"]:
                print(f"Parsing Counter ({counter['id']}): {counter['name']}")
                self.counters.append(Polygon(np.array(counter["coordinates"], np.int32)))

    def to_list(self):
        return self.counters

    def draw(self, frame, color):
        counter_img = np.zeros_like(frame, np.uint8)
        for counter in self.counters:
            counter_img = cv2.fillPoly(counter_img, [np.array(counter.exterior.coords, dtype=np.int32)], color)

        counter_mask = counter_img.astype(bool)
        frame[counter_mask] = cv2.addWeighted(frame, 0.2, counter_img, 0.8, 0)[counter_mask]
        return frame
