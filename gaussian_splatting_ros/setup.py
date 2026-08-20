import os
from glob import glob
from setuptools import setup

package_name = 'gaussian_splatting_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arena',
    maintainer_email='arena@example.com',
    description='ROS 2 node for asynchronous 3D Gaussian Splatting',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'splatting_node = gaussian_splatting_ros.node:main'
        ],
    },
)
