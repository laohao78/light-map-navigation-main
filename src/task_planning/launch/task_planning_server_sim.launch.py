from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('task_planning')
    
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'task_planning_sim.yaml'),
        description='Path to the parameters file'
    )
    
    osm_dir = get_package_share_directory('utils_pkg')
    osm_file_path = os.path.join(
        osm_dir,
        'resource',
        'osm',
        'medium.osm'
    )

    task_planning_node = Node(
        package='task_planning',
        executable='task_planning_server',
        name='task_planning_server',
        parameters=[
            LaunchConfiguration('params_file'),
            {'osm_file_path': osm_file_path}
        ],
        output='screen'
    )

    return LaunchDescription([params_file_arg, task_planning_node])