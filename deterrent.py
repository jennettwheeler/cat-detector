import time


def deter(position):
    print(f"DETERRENT!!!! {position}")


class Deterrent:
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self.last_time = time.time()

    def try_deter(self, position):
        current_time = time.time()
        if current_time - self.last_time >= self.min_interval:
            self.last_time = current_time
            deter(position)
