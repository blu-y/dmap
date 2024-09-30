import os
notebook_dir = os.path.dirname(os.path.abspath('__file__'))
print(notebook_dir)
os.chdir(notebook_dir)


import cv2
import yaml
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# os.chdir('/home/jetson/cmap/athirdmapper/exp01_0610')
os.chdir('./exp01_0610')
with open('features.pkl', 'rb') as file:
    features = pickle.load(file)
with open('features_ind.pkl', 'rb') as file:
    features_ind = pickle.load(file)
with open('text_features.pkl', 'rb') as file:
    text_features = pickle.load(file)
map_img = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
with open('map.yaml', 'r') as file:
    map_data = yaml.safe_load(file)

print(text_features['list'])
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



import cv2

# gt = np.copy(np.flip(_map, 0))
# gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
s = 5
# upscale the image
gt = cv2.resize(gt, (gt.shape[1]*s, gt.shape[0]*s), interpolation=cv2.INTER_NEAREST)
gt_ = np.copy(gt)
gtn = np.copy(gt)
_gt = np.copy(gt)
color = (0, 0, 255)
def draw_circle(event, x, y, flags, param):
    global color, s, gt_, gt
    x = x//(s*2)*(s*2)
    y = y//(s*2)*(s*2)
    if event == cv2.EVENT_LBUTTONDOWN:
        gt_ = np.copy(gt)
        for i in range(3*s):
            for j in range(3*s):
                if gt[y+i, x+j][0] == 0 and gt[y+i, x+j][1] == 0 and gt[y+i, x+j][2] == 0:
                    gt[y+i, x+j][0] = color[0]
                    gt[y+i, x+j][1] = color[1]
                    gt[y+i, x+j][2] = color[2]
        #cv2.rectangle(gt, (x, y), (x+(2*s), y+(2*s)), color, -1)
    if event == cv2.EVENT_RBUTTONDOWN:
        gt = np.copy(gt_)
cv2.namedWindow('Paint Image')
cv2.setMouseCallback('Paint Image', draw_circle)

legends = {
    'boxes': (0, 255, 0),
    'fire extinguisher': (0, 0, 255),
    'luggage': (255,0,0),
    'traffic cone': (0, 255, 255),
    'trash bin': (255, 255, 0),
    'umbrella': (255, 0, 255),
    'trolley': (128, 128, 0),
    'folded chair': (128, 0, 128),
}
n_legends = {
    'boxes': (0, 192, 255),
    'fire extinguisher': (0, 0, 192),
    'luggage': (196, 114, 68),
    'traffic cone': (49, 125, 237),
    'trash bin': (145, 133, 185),
    'umbrella': (230, 195, 157),
    'trolley': (117, 172, 152),
    'folded chair': (71,173,112),
    'white board': (124, 124, 124),
    'desk' : (116,114,64),
    'chair': (175, 165, 125)
}

while True:
    cv2.imshow('Paint Image', gt)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('w'):
        gt = np.copy(_gt)
    if key == ord('r'):
        gt = np.copy(gtn)
    if key == ord('c'):
        inst = input('Enter Instance:')
        try: color = n_legends[inst]
        except: color = (0, 0, 255)
        print(inst, color)
        _gt = np.copy(gt)
        continue
cv2.destroyAllWindows()
# downscale the image
gt = cv2.resize(gt, (gt.shape[1]//s, gt.shape[0]//s), interpolation=cv2.INTER_NEAREST)
gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
plt.imshow(gt, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()



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
    if torch.cuda.is_available(): 
        return image_features @ text_features.cpu().numpy().T
    return np.dot(image_features, text_features.T)
def show_images(ind, text='candidates'):
    for i, img_ind in enumerate(ind):
        r = len(ind)//7+1
        plt.figure(text, figsize=(9, r*2))
        plt.subplot(r, 7, i+1)
        img = cv2.imread('n_images/' + str(img_ind) + '.png')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
    plt.show()
def point2pixel(point, origin, resolution):
    return (int((point[0] - origin[0]) / resolution),
            int((point[1] - origin[1]) / resolution))
def probmap(conf, map, map_data, grid_size, mode='freq'):
    d = 0 if mode == 'score' else 1
    origin = map_data['origin']
    resolution = map_data['resolution']
    max_p = max(conf.values(), key=lambda x: x[d])[d]
    min_p = min(conf.values(), key=lambda x: x[d])[d]
    grid_pixel = int(grid_size / resolution // 2)
    ret = np.copy(map)
    ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
    color = (0,0,255)
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
        for point in features_ind[index]:
            [s, n] = conf.get(tuple(point), [0,0])
            conf[tuple(point)] = [(s * n + score) / (n + 1), n + 1]
    print(f'{m} frames detected')
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



    text_list = text_features['list']
for text in text_list:
    print(f'{text}')
    text_feature = text_features[text]
    conf = get_conf(features, features_ind, text_feature, show_candidates=True, text=text)
    probability_map = probmap(conf, _map, map_data, 0.25, 'freq')
    plt.figure(text)
    plt.axis('off')
    plt.tight_layout()
    probability_map = np.flip(probability_map, 0)
    probability_map = cv2.cvtColor(probability_map, cv2.COLOR_BGR2RGB)
    plt.imshow(probability_map)



os.chdir('../')
from utils import CLIP
os.chdir('../')
clip = CLIP('ViT-B-16-SigLIP')
os.chdir('./athirdmapper/exp01_0610')



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