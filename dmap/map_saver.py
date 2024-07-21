import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import yaml
import os
from nav_msgs.msg import OccupancyGrid
from .utils import maps_dir
import datetime

class MapSaver(Node):
    def __init__(self, fn='map'):
        super().__init__('map_saver')
        self.get_logger().info('Getting map data...')
        self.fd = maps_dir
        if not os.path.exists(self.fd): os.makedirs(self.fd)
        self.fn = fn
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.subscription  # Prevent unused variable warning
        self.map = OccupancyGrid()
        self.timer = self.create_timer(10, self.save_map)

    def map_callback(self, msg):
        self.map = msg
    
    def save_map(self):
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
            self.timer.cancel()
            raise SystemExit

        except Exception as e:
            self.get_logger().debug(f'Error: {e}') 
            self.get_logger().info('Failed to get map data, retrying...')
            pass

def main():
    rclpy.init()
    fn = datetime.datetime.now().strftime("%y%m%d_%H%M")
    node = MapSaver(fn)
    try:
        rclpy.spin(node)
    except SystemExit: rclpy.logging.get_logger('map_saver').info(f'Map saved, shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
