from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'replanner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='WJH',
    maintainer_email='wjh_9696@163.com',
    description='Replanner',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_replanner_server = replanner.waypoint_replanner_server:main',
            'waypoint_replanner_client = replanner.waypoint_replanner_client:main',
        ],
    },
)
