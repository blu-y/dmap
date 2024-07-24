import sys
import os
import glob
import PIL
import PIL.Image
from math import cos, sin, inf
import cv2
import numpy as np
import time
import rclpy
import rclpy.logging
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan, PointCloud2, PointField, Image
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros.buffer import Buffer
from tf2_geometry_msgs import do_transform_point # to transform
from tf2_ros.transform_listener import TransformListener
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from dmap import exp_dir, similarity, softmax
import datetime
import pickle

sys.path.append(os.getcwd())

class DMAPNode(Node):
    def __init__(self, args):
        '''
        camera: int (if usb camera, ex) 0, 1, 2, ...)
                str (if topic, ex) '/camera/image_raw')
        '''
        super().__init__('dmap_node')
        self.get_logger().info(', '.join(f'{k}={v}' for k, v in vars(args).items()))
        self.predefined = args.predefined
        self.load_features(args.feature_dir)
        if not self.predefined: 
            self.set_camera(args.camera)
            self.get_logger().info(f'Camera set to {self.camera}, {args.camera}')
            self.get_logger().info(f'CLIP model initializing to {args.model}')
            from dmap import CLIP
            self.clip = CLIP(args.model)
        self.get_logger().info(f'CLIP model initialized')
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.scan = LaserScan()
        self.leaf_size = args.leaf_size
        self.n_div = args.div
        self.inference = args.inference
        self.n_frames = 0
        self.avgfps = 0
        # HFOV 67.983 deg
        self.scan_from = -103
        self.scan_to = 101
        self.min_range = 0.3
        self.max_range = 3.0
        self.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
        self.last_scan_time = time.time()
        self.last_frame_time = 0.0
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, 1)
        self.scan_sub
        self.scan_pub = []
        for i in range(self.n_div):
            self.scan_pub.append(self.create_publisher(PointCloud2, f'/scan_{i}', 1))
        self.goal_sub = self.create_subscription(
            String, '/goal_str', self.goal_cb, 1)
        self.goal_sub
        self.map_sub = self.create_subscription(
            Bool, 
            '/save_map_req',
            self.map_cb,
            1
        )
        self.map_sub
        self.goal_pub = self.create_publisher(
            PoseStamped, '/goal_pose', 1)
        self.voxel_T = None
        self.debug = args.debug
        self.fd = os.path.join(exp_dir, datetime.datetime.now().strftime('%y%m%d_%H%M'))
        if not os.path.exists(self.fd): os.makedirs(self.fd)
        if not os.path.exists(f'{self.fd}/frames') and self.debug: os.makedirs(f'{self.fd}/frames')
        self.show_prob = args.show_prob
        self.get_logger().info('DMAP Node initialized')
        # TODO: to subscribe /save_map_req while processing scan

    def load_features(self, feature_dir):
        if feature_dir == '':
            self.features = []
            self.features_vox = []
            if self.predefined: raise Exception('Predefined text list is not supported without feature_dir')
        else:
            try:
                self.get_logger().info(f'Loading features from {feature_dir}')
                with open(os.path.join(feature_dir, 'features.pkl'), 'rb') as f:
                    self.features = pickle.load(f)
                with open(os.path.join(feature_dir, 'features_vox.pkl'), 'rb') as f:
                    self.features_vox = pickle.load(f)
                self.get_logger().info(f'Loaded {len(self.features)} features and {len(self.features_vox)} features_vox')
            except FileNotFoundError:
                self.get_logger().warn(f'\'features.pkl\' and \'features_vox.pkl\' not found in {feature_dir}, starting with empty features')
                self.features = []
                self.features_vox = []
            if self.predefined: 
                try:
                    with open(os.path.join(feature_dir, 'text_features.pkl'), 'rb') as f:
                        self.text_features = pickle.load(f)
                except FileNotFoundError:
                    raise Exception('\'text_features.pkl\' not found, please check the directory or run without \'--predefined\'')

    def map_cb(self, msg):
        self.save_features()

    def get_goal(self, text):
        if len(self.features) == 0: 
            self.get_logger().warn('No features to search goal')
            return None
        f_ind = self.features_vox
        if self.predefined:
            if text not in self.text_features:
                self.get_logger().warn(f'\'{text}\' not in text_features')
                self.get_logger().warn(f'Available texts: {list(self.text_features.keys())}')
                return None
            text_encodings = self.text_features[text]
            sim = similarity(self.features, text_encodings).squeeze()*100
            ssim = softmax(sim)
        else:
            try:
                text_encodings = self.clip.encode_text([text])
                sim = self.clip.similarity(self.features, text_encodings).squeeze()*100
                # ssim = F.softmax(torch.tensor(sim), dim=0).numpy()
                ssim = softmax(sim)
            except Exception as e:
                self.get_logger().warn(f'Failed to get similarity: {e}')
                return None
        nsim = np.zeros_like(ssim)
        nsim[ssim > 0.001] = 1
        m = int(np.sum(nsim))
        sim_sort_ind = np.argsort(sim, axis=0)[::-1]
        sim_sort_ind = sim_sort_ind[:m]
        conf = {}
        for index in sim_sort_ind:
            # print(len(features_vox[index]), features_vox[index])
            s_i = sim[index]
            n_point = len(f_ind[index])
            for point in f_ind[index]:
                [_s, _n] = conf.get(tuple(point), [0,0])
                conf[tuple(point)] = [(_s * _n + s_i) / (_n + 1), _n + 1]
        # sort confidence by value
        conf_score = dict(sorted(conf.items(), key=lambda item: item[1], reverse=True))
        conf_freq = dict(sorted(conf.items(), key=lambda item: item[1][1], reverse=True))
        ks = list(conf_score.keys())
        vs = list(conf_score.values())
        kf = list(conf_freq.keys())
        vf = list(conf_freq.values())
        [x, y, z] = kf[0]
        self.get_logger().info(f'\'{text}\' Goal: ({x:.2f}, {y:.2f}, {z:.2f}), score: {vf[0][0]:.3f}, freq: {vf[0][1]}')
        if self.debug:
            self.get_logger().debug(f'Keys in conf:\n\t{ks[:min(5, len(ks))]}\n\t{vs[:min(5, len(vs))]}')
            self.get_logger().debug(f'Keys in freq:\n\t{kf[:min(5, len(kf))]}\n\t{vf[:min(5, len(vf))]}')
        # TODO: get w
        return x, y, 1.0#w

    def goal_cb(self, msg):
        self.get_logger().info("Received goal command: '"+msg.data+"'")
        text = msg.data
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        try:
            goal.pose.position.x, goal.pose.position.y, goal.pose.orientation.w = self.get_goal(text)
        except: 
            self.get_logger().warn('Failed to get goal')
            return
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = 0.0
        self.goal_pub.publish(goal)

    def set_camera(self, camera):
        if isinstance(camera, int):
            self.get_logger().info(f'Using USB camera {camera}')
            self.camera = 'usb'
            from dmap import Camera
            self.cap = Camera(camera)
        elif isinstance(camera, str):
            if os.path.exists(os.path.expanduser(camera)):
                self.get_logger().info(f'Using local images {camera}')
                self.camera = 'local'
                self.image_list = glob.glob(os.path.join(camera, '*.png'))
                self.image_list.sort()
                self.get_logger().info(f'Found {len(self.image_list)} images')
                self.frame = PIL.Image.open(self.image_list[0])
                self.last_utc = os.path.basename(self.image_list[0]).rsplit('.',1)[0]
            else:
                self.get_logger().info(f'Using ROS camera topic {camera}')
                self.camera = 'ros'
                self.bridge = CvBridge() 
                self.camera_subscription = self.create_subscription(
                    Image,
                    camera,
                    self.camera_cb,
                    1
                )
                self.camera_subscription

    def get_frame(self, stamp):
        stamp = stamp.sec + stamp.nanosec * 1e-9
        if self.camera == 'usb':
            self.frame = self.cap.getFrame()
        elif self.camera == 'local':
            while True:
                curr_ = os.path.basename(self.image_list[0]).rsplit('.',1)[0]
                next_ = os.path.basename(self.image_list[1]).rsplit('.',1)[0]
                if float(next_) > stamp: break
                self.image_list.pop(0)
            self.frame = PIL.Image.open(self.image_list[0])
            stamp = float(curr_)
        # self.get_logger().debug(f'Requested time: {stamp}, Frame time: {curr_}')
        return self.frame, stamp

    def split_frame(self, frame):
        frame_div = []
        if isinstance(frame, PIL.Image.Image):
            w = frame.width
            h = frame.height
            for i in reversed(range(self.n_div)):
                frame_div.append(frame.crop((w//self.n_div*i, 0, w//self.n_div*(i+1), h)))
        if isinstance(frame, np.ndarray):
            w = frame.shape[1]
            for i in reversed(range(self.n_div)):
                frame_div.append(frame[:,w//self.n_div*i:w//self.n_div*(i+1),:])
        return frame_div

    def is_keyframe(self):
        ### TODO: Check if the frame is keyframe
        return True

    def encode_frame(self, frame, voxel_div):
        frame_div = self.split_frame(frame)
        if self.debug:
            for i in range(self.n_div):
                # frame_div[i].save(f'{self.fd}/frames/{len(self.features)+i}.png')
                cv2.imwrite(f'{self.fd}/frames/{len(self.features)+i}.png', frame_div[i])
        features = (self.clip.encode_images(frame_div))
        # features: [n_div] x [dim]
        self.features += features
        self.features_vox += voxel_div
        # self.features: [n_div * frames] x [dim]
        # self.features_vox: [n_div * frames] x [n_points] x [3]

    def scan_cb(self, msg: LaserScan):
        self.scan = msg
        if self.predefined or self.inference: return
        if self.clip.available: self.process_scan()

    def camera_cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def divide_ranges(self, points):
        size = len(points)//self.n_div
        ranges_div = [points[i*size:(i+1)*size] for i in range(self.n_div)]
        # points[:size], points[size:size*2], points[size*2:]
        return ranges_div

    def voxelize(self, points_div):
        voxel_div = []
        for i, points in enumerate(points_div):
            array = np.array(points, dtype=np.float32)
            quantized = np.round(array / self.leaf_size).astype(int)
            unique_voxels = np.unique(quantized, axis=0) * self.leaf_size
            voxel_div.append(unique_voxels.tolist())
        # voxel_div: [n_div] x [n_points] x [3]
        return voxel_div
    
    def transform_scan(self, scan: LaserScan):
        size = len(scan.ranges)//self.n_div
        # transform = self.tf_buffer.lookup_transform('map', 'base_link', scan.header.stamp)
        transform = self.tf_buffer.lookup_transform('map', scan.header.frame_id, rclpy.time.Time())
        ranges_div = self.divide_ranges(scan.ranges)
        points_div = []
        js = [i*size for i in range(self.n_div)]
        for ranges_i, j in zip(ranges_div,js):
            points = []
            for i, _range in enumerate(ranges_i):
                if _range == inf or _range == 'nan' or _range < self.min_range or _range > self.max_range: continue
                angle = scan.angle_min + (i+j+self.scan_from) * scan.angle_increment
                x = _range * cos(angle)
                y = _range * sin(angle)
                points_ = PointStamped()
                points_.header = scan.header
                points_.point.x = x
                points_.point.y = y
                # points_ = self.tf_buffer.transform(points_, 'map')
                points_ = do_transform_point(points_, transform)
                points.append([points_.point.x, points_.point.y, 0])
            points_div.append(points)
        return points_div

    def process_scan(self):
        try:
            start = time.time()
            scan = self.scan
            frame, stamp = self.get_frame(scan.header.stamp)
            if stamp == self.last_frame_time: return
            if not self.is_keyframe(): return
            self.last_frame_time = stamp
            if self.scan_from < 0:
                scan.ranges = scan.ranges[self.scan_from:] + scan.ranges[:self.scan_to]
            else: scan.ranges = scan.ranges[self.scan_from:self.scan_to]
            points_div = self.transform_scan(scan)
            header = scan.header
            header.frame_id = 'map'
            voxel_div = self.voxelize(points_div)
            self.encode_frame(frame, voxel_div)
            for i, voxel_i in enumerate(voxel_div):
                self.scan_pub[i].publish(pc2.create_cloud(header, self.fields, voxel_i))
            fps = 1/(time.time()-start)
            self.avgfps = (self.avgfps * self.n_frames + fps)/(self.n_frames+1)
            self.n_frames += 1
            self.get_logger().info(f"{[len(voxel_i) for voxel_i in voxel_div]} points added in feature {len(self.features_vox)-1}, {fps:.2f} fps, {self.avgfps:.2f} avgfps")
            self.last_scan_time = time.time()
            self.last_scan = scan
            self.last_frame = frame
        except KeyboardInterrupt: pass
        except Exception as e:
            self.get_logger().warn(f'{e}')
    
    def save_features(self):
        with open(f'{self.fd}/features.pkl', 'wb') as f:
            pickle.dump(self.features, f)
        with open(f'{self.fd}/features_vox.pkl', 'wb') as f:
            pickle.dump(self.features_vox, f)
        self.get_logger().info(f'Saved {len(self.features)} features and {len(self.features_vox)} features_vox in {self.fd}')

import argparse
from rclpy.executors import MultiThreadedExecutor

def camera_type(value):
    try:
        return int(value)
    except ValueError:
        return value

def main():
    parser = argparse.ArgumentParser(description='DMAP ROS2 Arguments')
    parser.add_argument('--camera', type=camera_type, default=0, help='Camera index or topic, directory')
    parser.add_argument('--model', type=str, default='ViT-B-16-SigLIP', help='Model name')
    parser.add_argument('--leaf_size', type=float, default=0.25, help='Leaf size for voxelization')
    parser.add_argument('--div', type=int, default=3, help='Number of divisions')
    parser.add_argument('--thread', type=int, default=4, help='Number of threads')
    parser.add_argument('--feature_dir', type=str, default='', help='Features directory')
    try: # python3.9+
        parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False, help='Debug mode')
        parser.add_argument('--show_prob', action=argparse.BooleanOptionalAction, default=False, help='Show probability')
        parser.add_argument('--inference', action=argparse.BooleanOptionalAction, default=False, help='Inference mode')
        parser.add_argument('--predefined', action=argparse.BooleanOptionalAction, default=False, help='Predefined text list mode')
    except: 
        parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='Debug mode')
        parser.add_argument('--show_prob', dest='show_prob', action='store_true', default=False, help='Show probability')
        parser.add_argument('--inference', dest='inference', action='store_true', default=False, help='Inference mode')
        parser.add_argument('--predefined', dest='predefined', action='store_true', default=False, help='Predefined text list mode')
    parser.add_argument('--ros-args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    rclpy.init(args=sys.argv)
    node = DMAPNode(args=args)
    rclpy.logging.set_logger_level('dmap_node', 
                                   rclpy.logging.LoggingSeverity.DEBUG if args.debug 
                                   else rclpy.logging.LoggingSeverity.INFO)
    if args.thread > 1:
        executer = MultiThreadedExecutor(num_threads=args.thread)
        executer.add_node(node)
    try:
        if args.thread > 1:
            executer.spin()
        else: rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as e:
        node.get_logger().error(f'Exception in executor spin: {e}')
    finally:
        node.save_features()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
