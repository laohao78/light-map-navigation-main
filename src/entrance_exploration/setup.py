from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'entrance_exploration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='WJH',
    maintainer_email='wjh_9696@163.com',
    description='Entrance exploration for the robot',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'entrance_exploration_action_server = entrance_exploration.entrance_exploration_action_server:main',
            'entrance_exploration_action_client = entrance_exploration.entrance_exploration_action_client:main',
        ],
    },
)
