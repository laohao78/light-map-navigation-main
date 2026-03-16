from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('entrance_exploration')
    
    config_file = os.path.join(
        pkg_dir,
        'config',
        'entrance_exploration_params_sim.yaml'
    )
    
    recognition_node = Node(
        package='building_entrance_recognition',
        executable='entrance_recognition_server',
        name='entrance_recognition_server',
        #namespace='entrance_exploration',
        output='screen'
    )
    
    exploration_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='entrance_exploration',
                executable='entrance_exploration_action_server',
                #namespace='entrance_exploration',
                parameters=[config_file],
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([
        recognition_node,
        exploration_node
    ])
