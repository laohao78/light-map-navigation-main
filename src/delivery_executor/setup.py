from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'delivery_executor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='WJH',
    maintainer_email='wjh_9696@163.com',
    description='Delivery Executor',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'delivery_executor_action_server = delivery_executor.delivery_executor_action_server:main',
            'delivery_executor_action_client = delivery_executor.delivery_executor_action_client:main',
        ],
    },
)
