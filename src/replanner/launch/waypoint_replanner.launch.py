from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_dir = get_package_share_directory('replanner')
    
    # Load the parameter file
    config_file = os.path.join(pkg_dir, 'config', 'waypoint_replanner.yaml')
    
    # Create and return launch description
    return LaunchDescription([
        Node(
            package='replanner',
            executable='waypoint_replanner_server',  # Make sure this matches the entry point name
            name='waypoint_replanner_server',
            parameters=[config_file],
            output='screen'
        )
    ])