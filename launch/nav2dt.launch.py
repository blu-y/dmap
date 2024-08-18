from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from dmap import example_dir
import os

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
        default_value='4',
        description='Number of threads'
    )
    feature_dir_arg = DeclareLaunchArgument(
        'feature_dir',
        default_value=example_dir,
        description='Features directory'
    )
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='',
        description='Debug mode'
    )
    show_prob_arg = DeclareLaunchArgument(
        'show_prob',
        default_value='',
        description='Show probability'
    )
    inference_arg = DeclareLaunchArgument(
        'inference',
        default_value='',
        description='Inference mode'
    )
    predefined_arg = DeclareLaunchArgument(
        'predefined',
        default_value='',
        description='Predefined text list mode'
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
            '--feature_dir', LaunchConfiguration('feature_dir'),
            # '--debug',
            '--show_prob',
            '--inference',
            '--predefined',
        ]
    )
    goal_server_node = Node(
        package='dmap',
        executable='goal_server',
        output='screen'
    )

    # Include the TurtleBot4 localization launch file
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare('turtlebot4_navigation').find('turtlebot4_navigation'),
                'launch',
                'localization.launch.py'
            )
        ),
        launch_arguments={'map': os.path.join(example_dir, 'map.yaml')}.items()
    )

    # Include the TurtleBot4 nav2 launch file
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare('turtlebot4_navigation').find('turtlebot4_navigation'),
                'launch',
                'nav2.launch.py'
            )
        ),
    )

    # Create and return the launch description
    return LaunchDescription([
        camera_arg,
        model_arg,
        leaf_size_arg,
        div_arg,
        thread_arg,
        feature_dir_arg,
        debug_arg,
        show_prob_arg,
        inference_arg,
        predefined_arg,
        dmap_node,
        goal_server_node,
        localization_launch,
        nav2_launch,
    ])

if __name__ == '__main__':
    generate_launch_description()
