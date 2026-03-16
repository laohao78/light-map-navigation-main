from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('building_entrance_recognition')
    
    # Create service node
    service_node = Node(
        package='building_entrance_recognition',
        executable='entrance_recognition_server',
        name='entrance_recognition_server',
        output='screen'
    )
    
    # Return launch description
    return LaunchDescription([
        service_node
    ]) 