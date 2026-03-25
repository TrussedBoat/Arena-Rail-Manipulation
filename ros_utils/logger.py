#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import numpy as np
import os
import cv2
from datetime import datetime


class JointCameraLogger(Node):
    def __init__(self):
        super().__init__('joint_camera_logger')
        self.bridge = CvBridge()

        # === Joint State Logging ===
        self.joint_names = []
        self.positions = []
        self.velocities = []
        self.efforts = []

        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        # === Camera Subscriptions ===
        self.image_topics = {
            "tripod_1": "/tripod_1/rgb",
            "tripod_2": "/tripod_2/rgb",
            "wrist": "/wrist/rgb",
            "tripod_1_d": "/tripod_1/depth",
            "tripod_2_d": "/tripod_2/depth",
            "wrist_d": "/wrist/depth"
        }

        self.latest_images = {}
        for cam_name, topic in self.image_topics.items():
            self.create_subscription(Image, topic, lambda msg, src=cam_name: self.image_callback(msg, src), 10)

        # === Setup Directories ===
        self.log_dir = os.path.join(os.getcwd(), 'log_data')
        os.makedirs(self.log_dir, exist_ok=True)
        self.image_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        # === Save Images Periodically ===
        self.create_timer(1.0, self.save_images)  # Every second

        self.get_logger().info("JointCameraLogger initialized.")

    def joint_callback(self, msg: JointState):
        self.joint_names.append(msg.name)
        self.positions.append(np.array(msg.position))
        self.velocities.append(np.array(msg.velocity))
        self.efforts.append(np.array(msg.effort))

    def image_callback(self, msg, source_name):
        try:
            encoding = msg.encoding.lower()

            if encoding in ["32fc1", "16uc1"]:
                # Depth image
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                norm = np.clip(depth_image, 0.0, 2.0)
                norm = (norm / 2.0 * 255).astype(np.uint8)
                image = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            else:
                # RGB image
                image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            self.latest_images[source_name] = image

        except Exception as e:
            self.get_logger().error(f"Error processing image from {source_name} (encoding={msg.encoding}): {e}")

    def save_images(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, img in self.latest_images.items():
            if img is not None:
                filename = os.path.join(self.image_dir, f"{name}_{timestamp}.png")
                # Convert RGB to BGR before saving
                if img.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img
                cv2.imwrite(filename, img_bgr)
                self.get_logger().info(f"Saved {filename}")



    def save_joint_data(self):
        np.save(os.path.join(self.log_dir, 'joint_names.npy'), self.joint_names)
        np.save(os.path.join(self.log_dir, 'positions.npy'), np.array(self.positions))
        np.save(os.path.join(self.log_dir, 'velocities.npy'), np.array(self.velocities))
        np.save(os.path.join(self.log_dir, 'efforts.npy'), np.array(self.efforts))
        self.get_logger().info(f"Saved joint state logs to {self.log_dir}")


def main(args=None):
    rclpy.init(args=args)
    node = JointCameraLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested. Saving joint state data...")
    finally:
        node.save_joint_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

