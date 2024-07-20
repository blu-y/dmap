import rclpy
from rclpy.executors import MultiThreadedExecutor
from cmapnode import CMAPNode

def main(args=None):
    # camera = 0
    # camera = '/camera/image_raw'
    camera = './athirdmapper/images/240622_1514'
    model = 'ViT-B-16-SigLIP'
    n_div = 3
    thread = 2
    rclpy.init(args=args)
    rclpy.logging.set_logger_level('cmap', rclpy.logging.LoggingSeverity.DEBUG)
    node = CMAPNode(model=model, camera=camera, n_div=n_div)
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