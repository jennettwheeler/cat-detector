import cv2
import queue
import threading
import time
import os.path


class LatestVideoCapture:
    fps = None
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)

        if os.path.isfile(name):
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            print('Playback FPS: '+str(self.fps))

        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            now = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

            diff = time.time() - now
            if self.fps and (diff < 1.0/(self.fps)):
                time.sleep(1.0/(self.fps) - diff)

    def read(self):
        try:
            return True, self.q.get(timeout=1)
        except queue.Empty:
            return False, None


    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()