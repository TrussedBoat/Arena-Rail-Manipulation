from langchain.tools import tool

import cv2
import numpy as np
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

import time
import base64


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

@tool
def publish_joint_command(position: list[float], name: list[str] = None, topic_name: str = '/sim/rail_franka1/joint_command') -> str:
    """
    Publishes a joint command to a ROS 2 topic.
    
    Args:
        position: List of target joint positions (e.g., in meters for prismatic joints like rail_j1).
        name: List of joint names. Defaults to ['rail_j1'].
        topic_name: The ROS 2 topic to publish to. Defaults to '/sim/rail_franka1/joint_command'.
    """
    if name is None:
        name = ['rail_j1']
        
    if not rclpy.ok():
        rclpy.init()
    
    node = rclpy.create_node('joint_command_publisher_tool')
    pub = node.create_publisher(JointState, topic_name, 10)
    
    # Wait for discovery so the message isn't dropped
    start_wait = time.time()
    while pub.get_subscription_count() == 0 and (time.time() - start_wait) < 2.0:
        time.sleep(0.1)
    
    time.sleep(0.1) # Brief buffer after discovery
    
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = name
    msg.position = [float(p) for p in position]
    
    pub.publish(msg)
    
    # Spin slightly to ensure message is published
    rclpy.spin_once(node, timeout_sec=1)
    node.destroy_node()

    print("Tool called to publish a joint command!")
    
    return f"Successfully published joint command: positions={position} to joints={name} on topic {topic_name}"
