import os
notebook_dir = os.path.dirname(os.path.abspath('__file__'))
print(notebook_dir)
os.chdir(notebook_dir)


from utils import CLIP
import os
import pickle
os.chdir('../')
clip = CLIP('ViT-L-14-quickgelu')
text_features = {}
text_list = ['boxes', 'fire extinguisher', 'luggage', 'a single traffic cone', 'pile of traffic cones', 'trash bin', 'umbrella', 'trolley', 'chair', 'folded chair']
text_features['list'] = text_list
for text in text_list:
    text_features[text] = clip.encode_text([text])
os.chdir('./athirdmapper')
with open('text_features.pkl', 'wb') as f:
    pickle.dump(text_features, f)
print(text_list)


import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
def similarity(image_features, text_features):
    if torch.cuda.is_available(): 
        return image_features @ text_features.cpu().numpy().T
    return image_features @ text_features.numpy().T
def show_candidates(ind):
    for i, img_ind in enumerate(ind):
        r = len(ind)//9+1
        plt.figure('candidates', figsize=(12, r*2))
        plt.subplot(r, 9, i+1)
        img = cv2.imread('n_images/' + str(img_ind) + '.png')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
with open('features.pkl', 'rb') as file:
    features = pickle.load(file)
with open('features_ind.pkl', 'rb') as file:
    features_ind = pickle.load(file)
with open('text_features.pkl', 'rb') as file:
    text_features = pickle.load(file)
print(text_features['list'])


text_feature = text_features['umbrella']
similarities = similarity(features, text_feature).squeeze()*100
# get softmax
similarities_s = F.softmax(torch.tensor(similarities), dim=0).numpy()
similarities_s
M_s = np.max(similarities_s)
m_s = np.min(similarities_s)
nsim = np.zeros_like(similarities_s)
nsim[similarities_s > 0.01] = 1
m = int(np.sum(nsim))
# nsim[similarities_s < 0.01] = 0
# n = int(np.sum(nsim))
print(M_s, m_s, m)

sim_sort_ind = np.argsort(similarities, axis=0)[::-1]
sim_sort_ind = sim_sort_ind[:m]
show_candidates(sim_sort_ind)


# print(sim_sort_ind)
# print(similarities_s[sim_sort_ind])
# get unique points
conf = {}
for index in sim_sort_ind:
    # print(len(features_ind[index]), features_ind[index])
    similarities_s = similarities[index]
    n_point = len(features_ind[index])
    for point in features_ind[index]:
        [s, n] = conf.get(tuple(point), [0,0])
        conf[tuple(point)] = [(s * n + similarities_s) / (n + 1), n + 1]
# sort confidence by value
conf_score = dict(sorted(conf.items(), key=lambda item: item[1], reverse=True))
conf_freq = dict(sorted(conf.items(), key=lambda item: item[1][1], reverse=True))

k = list(conf_score.keys())
v = list(conf_score.values())
print(k)
print(v)

K = list(conf_freq.keys())
V = list(conf_freq.values())
print(K)
print(V)