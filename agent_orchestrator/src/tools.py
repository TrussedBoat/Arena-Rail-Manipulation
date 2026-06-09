from langchain.tools import tool

import cv2
import numpy as np
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from scripts.arena.config import DEVICES, CAMERA_TOPICS

import time
import base64


class BackgroundROSBridge(Node):
    """A permanent ROS 2 node that caches state in the background."""
    def __init__(self):
        super().__init__('langgraph_background_bridge')
        self.bridge = CvBridge()
        
        # Caches
        self.latest_image = None
        self.rail_franka1_latest_joints = {} # Dictionary mapping joint names to positions
        
        # Permanent Subscribers
        self.create_subscription(Image, CAMERA_TOPICS["wall_franka_side"], self.image_cb, 10)
        self.create_subscription(JointState, DEVICES["rail_franka1"]["state_topic"], self.rail_franka1_joint_cb, 10)
        
        # Permanent Publisher
        self.rail_franka1_joint_pub = self.create_publisher(JointState, DEVICES["rail_franka1"]["command_topic"], 10)

    def image_cb(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def rail_franka1_joint_cb(self, msg):
        # Update the dictionary with the latest positions for all tracked joints
        for i, name in enumerate(msg.name):
            if name == DEVICES["rail_franka1"]["joint"]:
                self.rail_franka1_latest_joints[name] = msg.position[i]
                break


class ImageSubscriber(Node):
    def __init__(self, topic_name):
        super().__init__('get_image_tool')
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.latest_image = None

    def listener_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


def get_image_ros2(task: str, topic_name: str = '/sim/wall_franka/cam/front/color/image_raw') -> dict:
    """
    Subscribes to a ROS 2 image topic, saves the image to 'state.png', and returns the path.
    
    Args:
        topic_name (str): The ROS 2 topic to subscribe to. Defaults to '/sim/wall_franka/cam/front/color/image_raw'.

    Returns:
        dict: Dictionary containing 'text' and 'image' path.
    """
    image_path = "state.png"
    
    if os.path.exists(image_path):
        os.remove(image_path)

    if not rclpy.ok():
        rclpy.init()
    
    node = ImageSubscriber(topic_name)
    
    # Wait for one image
    start_time = time.time()
    timeout = 1.0 # 1 seconds timeout
    while node.latest_image is None and (time.time() - start_time) < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    
    image = node.latest_image
    node.destroy_node()
    
    if image is None:
        return f"Error: Timeout waiting for image on topic {topic_name}"
    
    # Save the image to disk
    success = cv2.imwrite(image_path, image)
    
    if not success:
        return "Error: Failed to save image to disk"

    print("Tool called to get an image!")

    return {"text": task, "image": os.path.abspath(image_path)}