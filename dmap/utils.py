import sys
import os
sys.path.append(os.getcwd())
import cv2
import numpy as np
import PIL
import time
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
import threading
from threading import Lock


class CLIP:
    def __init__(self, model='ViT-B-32'):
        pt = './'+model+'/open_clip_pytorch_model.bin'
        if torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')
        self.model, self.preprocess = create_model_from_pretrained(model, pretrained=pt, device=self.device)
        self.tokenizer = get_tokenizer(model)
        self.dim = 1
        self.encode_image(np.zeros((224, 224, 3), dtype=np.uint8))

    def encode_image(self, image):
        self.available = False
        if isinstance(image, np.ndarray): image = PIL.Image.fromarray(image)
        with torch.no_grad(), torch.cuda.amp.autocast():
            try: 
                image = self.preprocess(image).unsqueeze(0)
                if self.cuda: image = image.to(self.device, dtype=torch.float16)
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features[0].tolist()
            except Exception as e:
                print(e)
                image_features = [0.0] * self.dim
                pass
        self.available = True
        return image_features

    def encode_images(self, images):
        self.available = False
        new_images = []
        for image in images:
            if isinstance(image, np.ndarray): 
                new_images.append(PIL.Image.fromarray(image))
            else: new_images.append(image)
        images = new_images
        with torch.no_grad(), torch.cuda.amp.autocast():
            try:
                images = torch.stack([self.preprocess(image) for image in images])
                if self.cuda: images = images.to(self.device, dtype=torch.float16)
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.tolist()
            except Exception as e:
                print(e)
                image_features = [[0.0] * self.dim] * len(images)
                pass
        self.available = True
        return image_features

    def encode_text(self, label_list):
        text = self.tokenizer(label_list, context_length=self.model.context_length).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(): text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.tolist()
        if len(label_list) == 1: text_features = text_features[0]
        return text_features

    # def similarity(self, image_features, text_features):
    #     if self.cuda: return image_features @ text_features.cpu().numpy().T
    #     else: return image_features @ text_features.numpy().T

    def similarity(image_features, text_features):
        if isinstance(image_features, list): image_features = np.array(image_features)
        if isinstance(text_features, list): text_features = np.array(text_features)
        if torch.cuda.is_available(): 
            return image_features @ text_features.cpu().numpy().T
        return np.dot(image_features, text_features.T)

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
        
def make_utc(filename):
    filename = os.path.basename(filename).split('.')[0]
    yymmdd, hhmmss, ms = filename.split('_')
    utc = time.mktime(time.strptime(yymmdd+hhmmss, '%y%m%d%H%M%S'))
    return float(str(int(utc)) + '.' + ms)

from ament_index_python.packages import get_package_prefix, PackageNotFoundError

def find_ros2_package_src(package_name):
    try: package_path = get_package_prefix(package_name)
    except PackageNotFoundError: return None
    return package_path + '/../../src/' + package_name