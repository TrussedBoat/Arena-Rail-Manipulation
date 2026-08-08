#!/usr/bin/env python3
"""Display the simulated Panda wrist camera in an OpenCV window."""

import argparse

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class WristCameraViewer(Node):
    def __init__(self, color_topic: str, depth_topic: str, window: str, scale: float = 1.0) -> None:
        super().__init__("wrist_camera_viewer")
        self.window = window
        self.scale = scale
        self.bridge = CvBridge()
        self.view_mode = "color"
        
        self.color_sub = self.create_subscription(
            Image,
            color_topic,
            self.color_callback,
            10,
        )
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10,
        )
        self.get_logger().info(f"Viewing color: {color_topic}")
        self.get_logger().info(f"Viewing depth: {depth_topic}")
        self.get_logger().info("Controls: 'e' to toggle color/depth, 'q' or Esc to quit.")

    def _process_and_show(self, frame, is_depth=False):
        if self.scale != 1.0:
            width = int(frame.shape[1] * self.scale)
            height = int(frame.shape[0] * self.scale)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        if is_depth:
            # Replace NaNs and Infs with 0
            frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize to 0-255 dynamically for visibility
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Invert so closer objects (smaller depth) are white (255) and further are black (0)
            frame = 255 - frame

        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()
        elif key == ord('e'):
            if self.view_mode == "color":
                self.view_mode = "depth"
                self.get_logger().info("Switched to DEPTH view")
            else:
                self.view_mode = "color"
                self.get_logger().info("Switched to COLOR view")

    def color_callback(self, message: Image) -> None:
        if self.view_mode != "color":
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not convert color image: {exc}")
            return
        self._process_and_show(frame, is_depth=False)

    def depth_callback(self, message: Image) -> None:
        if self.view_mode != "depth":
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not convert depth image: {exc}")
            return
        self._process_and_show(frame, is_depth=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/sim/rail_franka1/cam/wrist/color/image_raw",
        help="ROS Color Image topic to display",
    )
    parser.add_argument(
        "--depth-topic",
        default="/sim/rail_franka1/cam/wrist/depth/image_raw",
        help="ROS Depth Image topic to display",
    )
    parser.add_argument("--window", default="Panda wrist camera")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor to resize the image/window (default: 0.5)",
    )
    args = parser.parse_args()

    rclpy.init()
    viewer = WristCameraViewer(args.topic, args.depth_topic, args.window, args.scale)
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
