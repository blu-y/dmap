import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import yaml
import os
import glob
# from dmap_msgs.srv import DmapGoal
from std_msgs.msg import String
from .utils import exp_dir

class GoalServer(Node):
    def __init__(self, fn=None):
        super().__init__('goal_server')
        self.pub = self.create_publisher(
            String,
            '/goal_str',
            10
        )
        # self.srv = self.create_service(DmapGoal, '/goal_command', self.command_cb)
        
    def command_cb(self, request, response):
        try:
            req_str = request.command
            self.get_logger().info('Received goal command: '+req_str)
            msg = String()
            msg.data = req_str
            self.pub.publish(msg)
            response.result = True
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
            response.result = False

def main(args=None):
    rclpy.init()
    node = GoalServer()
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
