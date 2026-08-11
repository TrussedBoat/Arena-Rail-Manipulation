from glob import glob
from setuptools import find_packages, setup


package_name = "semantic_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="homerobotics",
    maintainer_email="romerhomerobotics@gmail.com",
    description="RGB-D semantic perception and persistent object registry",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "semantic_perception_node = semantic_perception.node:main",
            "yolo_debug_node = semantic_perception.yolo_debug:main",
            "semantic_rviz_visualizer = semantic_perception.rviz_visualizer:main",
            "semantic_rviz_visualizer = semantic_perception.rviz_visualizer:main",
        ],
    },
)
