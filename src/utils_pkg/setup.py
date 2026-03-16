from setuptools import find_packages, setup
from glob import glob

package_name = 'utils_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource/osm', glob('resource/osm/*')),
    ],
    install_requires=['setuptools',
                      'requests',
                      'Pillow',
                      'opencv-python',
                      'numpy',
                      ],
    zip_safe=True,
    maintainer='WJH',
    maintainer_email='wjh_9696@163.com',
    description='Utils for this projects',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'osm2pgm = utils_pkg.osm2pgm:main',
            'transform_pgm = utils_pkg.transform_pgm:main',
            'osm_postprocessing = utils_pkg.osm_postprocessing:main',
        ],
    },
)
