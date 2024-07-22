from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare the launch arguments
    camera_arg = DeclareLaunchArgument(
        'camera',
        default_value='0',
        description='Camera index or topic, directory'
    )
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='ViT-B-16-SigLIP',
        description='Model name'
    )
    leaf_size_arg = DeclareLaunchArgument(
        'leaf_size',
        default_value='0.25',
        description='Leaf size for voxelization'
    )
    div_arg = DeclareLaunchArgument(
        'div',
        default_value='3',
        description='Number of divisions'
    )
    thread_arg = DeclareLaunchArgument(
        'thread',
        default_value='2',
        description='Number of threads'
    )
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Debug mode'
    )
    show_prob_arg = DeclareLaunchArgument(
        'show_prob',
        default_value='false',
        description='Show probability'
    )
    feature_dir_arg = DeclareLaunchArgument(
        'feature_dir',
        default_value='',
        description='Features directory'
    )
    inference_arg = DeclareLaunchArgument(
        'inference',
        default_value='false',
        description='Inference mode'
    )

    # Define the node with configurations
    dmap_node = Node(
        package='dmap',
        executable='dmap',
        output='screen',
        arguments=[
            '--camera', LaunchConfiguration('camera'),
            '--model', LaunchConfiguration('model'),
            '--leaf_size', LaunchConfiguration('leaf_size'),
            '--div', LaunchConfiguration('div'),
            '--thread', LaunchConfiguration('thread'),
            '--debug', LaunchConfiguration('debug'),
            '--show_prob', LaunchConfiguration('show_prob'),
            '--feature_dir', LaunchConfiguration('feature_dir'),
            '--inference', LaunchConfiguration('inference')
        ]
    )

    # Create and return the launch description
    return LaunchDescription([
        camera_arg,
        model_arg,
        leaf_size_arg,
        div_arg,
        thread_arg,
        debug_arg,
        show_prob_arg,
        feature_dir_arg,
        inference_arg,
        dmap_node
    ])
if __name__ == '__main__':
    generate_launch_description()
