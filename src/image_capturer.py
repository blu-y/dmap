import cv2
import datetime
import time
last_t = time.time()
def capture_image():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
    while True:
        ret, frame = cap.read()
        print(frame.shape)
        if not ret:
            print('Failed to capture image')
            break
        cv2.imshow('Camera', frame)
        # Check if the Enter key is pressed
        key = cv2.waitKey(1)
        if last_t - time.time() >= 0.1:
            current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")[:-4]
            file_name = f"./images/{current_time}.png"
            cv2.imwrite(file_name, frame)
            print(frame.shape, 'Image saved!', file_name)
            last_t = time.time()
        # Check if the Esc key is pressed
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
capture_image()