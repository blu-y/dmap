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
        self.fd = exp_dir
        if not os.path.exists(self.fd): os.makedirs(self.fd)
        self.fn = fn
        self.sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.pub = self.create_publisher(
            Bool,
            '/save_map_req',
            1
        )
        self.srv = self.create_service(Trigger, '/save_map', self.save_map_cb)
        self.sub  # Prevent unused variable warning
        self.map = None

    def map_callback(self, msg):
        self.map = msg
        self.get_logger().info(f'Map Updated: {msg.info.width}x{msg.info.height}')
    
    def save_map_cb(self, request, response):
        self.get_logger().info('Saving map...')
        if self.fn is None:
            self.fn = sorted(os.listdir(exp_dir))[-1]+'/map'
        if self.map is None:
            self.get_logger().info('No map data received yet')
            return
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
            pgm_fn = os.path.join(self.fd, self.fn+'.pgm')
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
            yaml_fn = os.path.join(self.fd, self.fn+'.yaml')
            with open(yaml_fn, 'w') as yaml_file:
                yaml.dump(map_metadata, yaml_file)
            self.get_logger().info(f'Map saved as {self.fn}.pgm and {self.fn}.yaml')
            response.success = True
            response.message = f'Map saved as {self.fn}.pgm and {self.fn}.yaml'
        except Exception as e:
            self.get_logger().debug(f'Error: {e}') 
            self.get_logger().info('Failed to get map data, retrying...')
            response.success = False

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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
