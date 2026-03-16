# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.substitutions import PathJoinSubstitution
# from launch_ros.substitutions import FindPackageShare, FindPackagePrefix
# import os

# def generate_launch_description():
#     config_file = PathJoinSubstitution([
#         FindPackageShare('grounded_sam2'),
#         'config',
#         'gsam2_params.yaml'
#     ])

#     package_path = str(FindPackagePrefix('grounded_sam2').find('grounded_sam2'))
#     custom_path = os.path.join(os.path.dirname(package_path), '../src/grounded_sam2/grounded_sam2/grounded_sam2')
    
#     current_pythonpath = os.environ.get('PYTHONPATH', '')
#     os.environ['PYTHONPATH'] = f"{custom_path}:{current_pythonpath}"
    
#     conda_prefix = os.environ.get('CONDA_PREFIX', '/opt/conda/envs/dl_env')
#     python_path = os.path.join(conda_prefix, 'bin', 'python')

#     return LaunchDescription([
#         Node(
#             package='grounded_sam2',
#             executable='gsam2_node',
#             name='grounded_sam2_node',
#             prefix=[python_path],
#             output='screen',
#             parameters=[config_file]
#         )
#     ])


#wwt
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare('grounded_sam2'),
        'config',
        'gsam2_params.yaml'
    ])

    return LaunchDescription([
        Node(
            package='grounded_sam2',
            executable='gsam2_node',
            name='grounded_sam2_node',
            output='screen',
            parameters=[config_file],
        )
    ])
