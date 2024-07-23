import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import PIL
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from .utils import models_dir

class CLIP:
    def __init__(self, model='ViT-B-32'):
        pt = os.path.join(models_dir, model, 'open_clip_pytorch_model.bin')
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
            except KeyboardInterrupt: pass
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
            except KeyboardInterrupt: pass
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

    def similarity(self, image_features, text_features):
        if isinstance(image_features, list): image_features = np.array(image_features)
        if isinstance(text_features, list): text_features = np.array(text_features)
        # if torch.cuda.is_available(): 
        #     return image_features @ text_features.cpu().numpy().T
        return np.dot(image_features, text_features.T)