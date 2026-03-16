from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    ipm_pkg_dir = get_package_share_directory('ipm_image_node')
    
    # Declare parameter file path argument
    ipm_config_file = os.path.join(
        ipm_pkg_dir,
        'config',
        'ipm_params.yaml'
    )

    # Create IPM node
    ipm_node = Node(
        package='ipm_image_node',
        executable='ipm',
        name='ipm_image_node',
        parameters=[ipm_config_file],
        output='screen'
    )

    return LaunchDescription([
        ipm_node
    ])