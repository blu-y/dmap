import os
notebook_dir = os.path.dirname(os.path.abspath('__file__'))
print(notebook_dir)



import cv2
import yaml
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
os.chdir(notebook_dir)
os.chdir('./exp0610_ViT-B-16-SigLIP_3_copy')
map_img = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
with open('map.yaml', 'r') as file:
    map_data = yaml.safe_load(file)

print(np.unique(map_img))
_map = np.copy(map_img)
_map[_map == 101] = 255
_map[_map == 100] = 220
plt.imshow(np.flip(_map, 0), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
# 101 = unknown space
# 0 = occupied space
# 100 = free space => 255



import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
def similarity(image_features, text_features):
    if isinstance(image_features, list): image_features = np.array(image_features)
    if isinstance(text_features, list): text_features = np.array(text_features)
    try: 
        if torch.cuda.is_available(): 
            return image_features @ text_features.cpu().numpy().T
    except: return np.dot(image_features, text_features.T)
def show_images(ind, text='candidates'):
    for i, img_ind in enumerate(ind):
        r = len(ind)//7+1
        plt.figure(text, figsize=(9, r*2))
        plt.subplot(r, 7, i+1)
        img = cv2.imread('n_images/' + str(img_ind) + '.png')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
    plt.savefig(f'{text}_cdd.png')
    plt.show()
def point2pixel(point, origin, resolution):
    return (int((point[0] - origin[0]) / resolution),
            int((point[1] - origin[1]) / resolution))
def probmap(conf, map, map_data, grid_size, mode='freq', show=True):
    d = 0 if mode == 'score' else 1
    origin = map_data['origin']
    resolution = map_data['resolution']
    grid_pixel = int(grid_size / resolution // 2)
    ret = np.copy(map)
    ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
    color = (0,0,255)
    if len(conf) == 0: return ret
    max_p = max(conf.values(), key=lambda x: x[d])[d]
    min_p = min(conf.values(), key=lambda x: x[d])[d]
    for point, score in conf.items():
        x, y = point2pixel(point, origin, resolution)
        try: p = (score[d]-min_p) / (max_p-min_p)
        except ZeroDivisionError: p = 1
        layer = np.copy(ret)
        layer = cv2.rectangle(layer, (x-grid_pixel,y-grid_pixel), (x+grid_pixel,y+grid_pixel), color, -1)
        ret = cv2.addWeighted(ret, 1-p, layer, p, 0)
    return ret
def get_conf(features, voxels, text_feature, th=0.01, show_candidates=False, text='candidates'):
    similarities = similarity(features, text_feature).squeeze()*100
    softmax = F.softmax(torch.tensor(similarities), dim=0).numpy()
    m = np.sum(softmax > th)
    sim_sort_ind = np.argsort(similarities, axis=0)[::-1][:m]
    conf = {}
    for index in sim_sort_ind:
        # score = similarities[index]
        score = softmax[index]
        for point in voxels[index]:
            [s, n] = conf.get(tuple(point), [0,0])
            conf[tuple(point)] = [(s * n + score) / (n + 1), n + 1]
    print(f'{m} frames detected')
    print(sim_sort_ind)
    print(f'{len(conf)} grid points are detected')
    print('conf:', conf)
    if show_candidates: show_images(sim_sort_ind, text=text)
    # sort confidence by value
    # conf_score = dict(sorted(conf.items(), key=lambda item: item[1], reverse=True))
    # conf_freq = dict(sorted(conf.items(), key=lambda item: item[1][1], reverse=True))
    # k_s = list(conf_score.keys())
    # v_s = list(conf_score.values())
    # k_f = list(conf_freq.keys())
    # v_f = list(conf_freq.values())
    return conf



os.chdir(notebook_dir)
os.chdir('./exp0610_ViT-B-16-SigLIP_3_copy')
with open('features.pkl', 'rb') as file:
    features = pickle.load(file)
with open('features_ind.pkl', 'rb') as file:
    features_ind = pickle.load(file)
with open('text_features.pkl', 'rb') as file:
    text_features = pickle.load(file)
print(text_features['list'])
text_list = text_features['list']
for text in text_list:
    print(f'{text}')
    text_feature = text_features[text]
    conf = get_conf(features, features_ind, text_feature, show_candidates=True, text=text)
    probability_map = probmap(conf, _map, map_data, 0.25, 'freq', show=True)
    plt.figure(text)
    plt.axis('off')
    plt.tight_layout()
    probability_map = np.flip(probability_map, 0)
    probability_map = cv2.cvtColor(probability_map, cv2.COLOR_BGR2RGB)
    plt.imshow(probability_map)
    plt.savefig(f'{text}_prob.png')





os.chdir(notebook_dir)
from utils import CLIP
os.chdir('../')
clip = CLIP('ViT-B-16-SigLIP')



os.chdir(notebook_dir)
os.chdir('./exp0610_ViT-B-16-SigLIP_5')
import time
with open('features.pkl', 'rb') as file:
    features = pickle.load(file)
with open('features_ind.pkl', 'rb') as file:
    features_ind = pickle.load(file)
with open('text_features.pkl', 'rb') as file:
    text_features = pickle.load(file)
print(text_features['list'])
text_list = text_features['list']
cumtime = 0
for text in text_list:
    print(f'{text}')
    start = time.time()
    text_feature = clip.encode_text([text])
    conf = get_conf(features, features_ind, text_feature, show_candidates=False, text=text)
    conf_freq = dict(sorted(conf.items(), key=lambda item: item[1][1], reverse=True))
    try: goal = list(conf_freq.keys())[0]
    except: goal = 'None'
    flowtime = time.time()-start
    print(f'goal: {goal}, query time: {flowtime:.2f}s')
    cumtime += flowtime
avgtime = cumtime/len(text_list)
print(f'avgtime: {avgtime:.3f}')

# ViT-B-16-SigLIP
# 1352 / 9.7 / 0.149
# 2516 / 17.8 / 0.205
# 3018 / 21.2 / 0.215
# 4156 / 29.1 / 0.251
# 4900 / 34.2 / 0.278

# ViT-B-32
# 1384 / 6.8 / 0.129
# 2480 / 11.8 / 0.166
# 3633 / 17.1 / 0.201
# 4704 / 22.1 / 0.256
# 5470 / 25.6 / 0.231

# ViT-B-32-256
# 1346 / 6.6 / 0.123
# 2616 / 12.4 / 0.170
# 3787 / 17.7 / 0.206
# 4961 / 23.2 / 0.248
# 5205 / 24.3 / 0.276

# ViT-L-14-quickgelu
# 1011 / 7.3MB / 0.127
# 1612 / 11.4MB / 0.175
# 2094 / 14.7MB / 0.196
# 2424 / 17.0MB / 0.201
# 2665 / 18.6MB / 0.225

os.chdir(notebook_dir)
from utils import CLIP
os.chdir('../')
clip = CLIP('ViT-B-16-SigLIP')
os.chdir(notebook_dir)
os.chdir('./exp0610_ViT-B-16-SigLIP_1')



from IPython.display import clear_output
while True:
    try:
        clear_output(wait=True)
        string = input('Enter text: ')
        if string == '': break
        print(f'{string}')
        text_feature = clip.encode_text([string])
        conf = get_conf(features, features_ind, text_feature, show_candidates=True, text=string)
        probability_map = probmap(conf, _map, map_data, 0.25, 'freq')
        plt.figure(string)
        plt.axis('off')
        plt.tight_layout()
        probability_map = np.flip(probability_map, 0)
        probability_map = cv2.cvtColor(probability_map, cv2.COLOR_BGR2RGB)
        plt.imshow(probability_map)
        plt.show()
    except Exception as e:
        print(e)