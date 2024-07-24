def make_utc(filename):
    import os
    import time
    filename = os.path.basename(filename).split('.')[0]
    yymmdd, hhmmss, ms = filename.split('_')
    utc = time.mktime(time.strptime(yymmdd+hhmmss, '%y%m%d%H%M%S'))
    return float(str(int(utc)) + '.' + ms)

from ament_index_python.packages import get_package_prefix, PackageNotFoundError
import os

def find_ros2_package_dir(package_name):
    try: package_path = get_package_prefix(package_name)
    except PackageNotFoundError: return None
    package_path = os.path.dirname(os.path.dirname(package_path))
    return os.path.join(package_path, 'src', package_name)

import numpy as np

def similarity(image_features, text_features):
    if isinstance(image_features, list): image_features = np.array(image_features)
    if isinstance(text_features, list): text_features = np.array(text_features)
    return np.dot(image_features, text_features.T)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

dmap_dir = find_ros2_package_dir('dmap')
models_dir = os.path.join(dmap_dir, 'models')
maps_dir = os.path.join(dmap_dir, 'maps')
exp_dir = os.path.join(dmap_dir, 'exp')
example_dir = os.path.join(dmap_dir, 'example')