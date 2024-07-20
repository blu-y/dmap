import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import yaml
import os
from nav_msgs.msg import OccupancyGrid

class MapSaver(Node):
    def __init__(self, fn='map.pgm'):
        super().__init__('map_saver')
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
            print(np.unique(map_data))
            
            # Convert the occupancy data to a PGM image
            pgm_data = np.full_like(map_data, 205, dtype=np.uint8)
            pgm_data[np.logical_and(map_data >= 0, map_data <= 25)] = 254
            pgm_data[np.logical_and(map_data >= 65, map_data <= 100)] = 0
            cv2.imwrite(self.fn+'.pgm', pgm_data)

            # Save the metadata to a YAML file
            map_metadata = {
                'image': f'{os.path.basename(self.fn)}.pgm',
                'resolution': resolution,
                'origin': [origin.position.x, origin.position.y, origin.position.z],
                'negate': 0,
                'occupied_thresh': 0.65,
                'free_thresh': 0.25
            }
            with open(self.fn+'.yaml', 'w') as yaml_file:
                yaml.dump(map_metadata, yaml_file)
            
            self.get_logger().info(f'Map saved as {self.fn}.pgm and {self.fn}.yaml')
        except: pass

def main():
    rclpy.init()
    fn = './athirdmapper/map'
    node = MapSaver(fn)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
