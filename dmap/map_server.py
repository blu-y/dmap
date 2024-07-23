import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import cv2
import yaml
import os
import glob
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger
from std_msgs.msg import Bool
from .utils import exp_dir

class MapServer(Node):
    def __init__(self, fn=None):
        super().__init__('map_server')
        self.get_logger().info('Getting map data...')
        if not os.path.exists(exp_dir): os.makedirs(exp_dir)
        if fn is None:
            self.fn = 'map'
        else: self.fn = fn
        self.sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.sub
        self.pub = self.create_publisher(
            Bool,
            '/save_map_req',
            1
        )
        self.srv = self.create_service(Trigger, '/save_map', self.save_map_cb)
        self.get_logger().info('Map Server Ready\n \
                                ros2 service call /save_map std_srvs/srv/Trigger')
        self.map = None

    def map_callback(self, msg):
        self.map = msg
        self.get_logger().info(f'Map Updated: {msg.info.width}x{msg.info.height}')
    
    def save_map_cb(self, request, response):
        self.get_logger().info('Saving map...')
        all_folders = [f for f in glob.glob(exp_dir + "/*") if os.path.isdir(f)]
        fd = max(all_folders, key=os.path.getctime)
        if self.map is None:
            self.get_logger().info('No map data received yet')
            response.success = False
            response.message = 'No map data received yet'
            return response
        msg = Bool()
        msg.data = True
        self.pub.publish(msg)
        try:
            msg = self.map
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution
            origin = msg.info.origin
            map_data = np.array(msg.data).reshape((height, width))
            self.get_logger().debug(f'{np.unique(map_data)}')
            # Convert the occupancy data to a PGM image
            pgm_data = np.full_like(map_data, 205, dtype=np.uint8)
            pgm_data[np.logical_and(map_data >= 0, map_data <= 25)] = 254
            pgm_data[np.logical_and(map_data >= 65, map_data <= 100)] = 0
            self.get_logger().debug(f'{np.unique(pgm_data)}')
            pgm_fn = os.path.join(fd, self.fn+'.pgm')
            cv2.imwrite(pgm_fn, pgm_data)
            # Save the metadata to a YAML file
            map_metadata = {
                'image': f'{self.fn}.pgm',
                'resolution': resolution,
                'origin': [origin.position.x, origin.position.y, origin.position.z],
                'negate': 0,
                'occupied_thresh': 0.65,
                'free_thresh': 0.25
            }
            yaml_fn = os.path.join(fd, self.fn+'.yaml')
            with open(yaml_fn, 'w') as yaml_file:
                yaml.dump(map_metadata, yaml_file)
            self.get_logger().info(f'Map saved as {pgm_fn} and {yaml_fn}')
            response.success = True
            response.message = f'Map saved as {pgm_fn} and {yaml_fn}'
        except Exception as e:
            self.get_logger().info(f'Failed to get map data, {e}')
            response.success = False
            response.message = f'Failed to get map data, {e}'
        return response

def main():
    rclpy.init()
    node = MapServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as e: 
        node.get_logger().error(f'Error: {e}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
