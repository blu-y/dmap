import cv2
import datetime
import time
import os
from dmap import dmap_dir
def capture_image(cam=0, w=1024, h=576, fps=30):
    last_t = time.time()
    cap = cv2.VideoCapture(cam, cv2.CAP_V4L)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    src = dmap_dir
    fd = datetime.datetime.now().strftime("%y%m%d_%H%M")
    fd = src + '/images/' + fd
    # print(fd)
    os.path.exists(f"{fd}") or os.makedirs(f"{fd}")
    while True:
        ret, frame = cap.read()
        # print(frame.shape)
        if not ret:
            print('Failed to capture image')
            break
        cv2.imshow('Camera', frame)
        # Check if the Enter key is pressed
        key = cv2.waitKey(1)
        fn = time.time()
        if fn - last_t >= 0.02:
            file_name = f"{fd}/{fn:.7f}.png"
            cv2.imwrite(file_name, frame)
            print(frame.shape, 'Image saved!', file_name)
            last_t = fn
        # Check if the Esc key is pressed
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
capture_image(2)