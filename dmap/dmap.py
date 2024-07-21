import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
from dmap import DMAPNode

def main(args=None):
    # camera: Int, String
    # Int for camera index, String for camera topic or directory
    # camera = 0
    # camera = '/camera/image_raw'
    # camera = './images/240622_1514'
    camera = 2
    model = 'ViT-B-16-SigLIP'
    n_div = 3
    thread = 2
    rclpy.init(args=args)
    node = DMAPNode(model=model, camera=camera, n_div=n_div, debug=True)
    rclpy.logging.set_logger_level('dmap_node', rclpy.logging.LoggingSeverity.DEBUG)
    if thread > 1:
        executer = MultiThreadedExecutor(num_threads=thread)
        executer.add_node(node)
    try:
        if thread > 1:
            executer.spin()
        else: rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()