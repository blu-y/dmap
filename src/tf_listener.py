#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Vector3, Quaternion
import time
import cv2

class TFListenerNode(Node):

    def __init__(self):
        super().__init__('tf_listener_node')
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )
        self.translation = Vector3()
        self.rotation = Quaternion()
        self.last_t = time.time()
        self.timer = self.create_timer(0.1, self.timer_cb)

    def tf_callback(self, msg):
        # Callback function to handle incoming TF messages
        # You can perform any desired operations here
        for t in msg.transforms:
            if t.header.frame_id == 'odom' and t.child_frame_id == 'base_link':
                s = t.header.stamp.sec
                ns = t.header.stamp.nanosec
                x = t.transform.translation.x
                y = t.transform.translation.y
                z = t.transform.translation.z
                qx = t.transform.rotation.x
                qy = t.transform.rotation.y
                qz = t.transform.rotation.z
                qw = t.transform.rotation.w
                if t.transform.translation != self.translation or t.transform.rotation != self.rotation:
                    self.translation = t.transform.translation
                    self.rotation = t.transform.rotation
                    # # Write the data to tf.txt file
                    # with open('/home/kau/image/tf.txt', 'a') as file:
                    #     file.write(f"{s}.{ns}, {x}, {y}, {z}, {qx}, {qy}, {qz}, {qw}\n")
                    self.get_logger().info(f"{s}.{ns}, {x}, {y}, {z}, {qx}, {qy}, {qz}, {qw}\n")

    def timer_cb(self):
        ret, frame = self.cap.read()
        if not ret:
            print('Failed to capture image')
            return
        # cv2.imshow('Camera', frame)
        # key = cv2.waitKey(1)
        current_time = time.time()
        if current_time - self.last_t >= 0.1:
            file_name = f"./images/{current_time}.png"
            with open('/home/kau/image/tf.txt', 'a') as file:
                file.write(f"{current_time}, {self.translation.x}, {self.translation.y}, {self.translation.z}, {self.rotation.x}, {self.rotation.y}, {self.rotation.z}, {self.rotation.w}\n")
            cv2.imwrite(file_name, frame)
            self.last_t = current_time
        # if key == 27:
        #     self.cap.release()
        #     cv2.destroyAllWindows()
        #     self.get_logger().info('Exiting...')
        #     rclpy.shutdown()
        #     exit()
            
def main(args=None):
    rclpy.init(args=args)
    tf_listener_node = TFListenerNode()
    rclpy.spin(tf_listener_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()