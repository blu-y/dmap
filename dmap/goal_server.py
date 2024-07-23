import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import yaml
import os
import glob
from dmap_msgs.srv import DmapGoal
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_msgs.msg import Bool
from .utils import exp_dir
import time

class GoalServer(Node):
    def __init__(self, fn=None):
        super().__init__('goal_server')
        self.pub = self.create_publisher(
            String,
            '/goal_str',
            10
        )
        self.srv = self.create_service(DmapGoal, '/goal_command', self.command_cb)
        self.get_logger().info('Goal Server Ready\n \
                               ros2 service call /goal_command dmap_msgs/srv/DmapGoal "{command: \'a bottle of water\'}"')

    def command_cb(self, request, response):
        try:
            req_str = request.command
            self.get_logger().info('Service: received command: \''+req_str+'\'')
            msg = String()
            msg.data = req_str
            self.pub.publish(msg)
            response.result = True
        except Exception as e:
            self.get_logger().error(f'Goal not published')
            response.result = False
        return response

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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
