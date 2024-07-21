import threading
from threading import Lock
import cv2

class Camera:
    last_frame = None
    last_ready = None
    lock = Lock()
    capture=None
    def __init__(self, rtsp_link, w=1024, h=576):
        self.w = w
        self.h = h
        self.capture = cv2.VideoCapture(rtsp_link)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(), name="rtsp_read_thread")
        thread.daemon = True
        thread.start()
    def rtsp_cam_buffer(self):
        while True:
            with self.lock:
                self.last_ready = self.capture.grab()
    def getFrame(self):
        if (self.last_ready is not None):
            self.last_ready,self.last_frame=self.capture.retrieve()
            return self.last_frame.copy()
        else:
            return -1