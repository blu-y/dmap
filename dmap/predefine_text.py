import argparse
from dmap import CLIP
import rclpy
from rclpy.node import Node
import pickle

# Usage: ros2 run dmap predefine_text.py -l <text1> <text2> ...

def main():
    parser = argparse.ArgumentParser("Predefine text list")
    parser.add_argument('-l','--list', nargs='+', help='<Required> text list', required=True)
    parser.add_argument('-m', '--model', type=str, default='ViT-B-16-SigLIP', help='Model name')
    parser.add_argument('-o', '--output', help='output file name', default=f'./text_features.pkl')
    args = parser.parse_args()
    rclpy.init()
    node = Node('predefine_text')
    node.get_logger().info(f'Generating text features for {args.list}')
    node.get_logger().info(f'Model: {args.model}')
    clip = CLIP(model=args.model)
    text_features = {}
    for text in args.list:
        text_features[text] = clip.encode_text([text])
    node.get_logger().info(f'{len(text_features)} text features are generated')
    with open(args.output, 'wb') as f:
        pickle.dump(text_features, f)
    node.get_logger().info(f'saved to {args.output}')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()