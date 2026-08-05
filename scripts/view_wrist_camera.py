#!/usr/bin/env python3
"""Display the simulated Panda wrist camera in an OpenCV window."""

import argparse

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class WristCameraViewer(Node):
    def __init__(self, topic: str, window: str, scale: float = 1.0) -> None:
        super().__init__("wrist_camera_viewer")
        self.window = window
        self.scale = scale
        self.bridge = CvBridge()
        self.frames_received = 0
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10,
        )
        self.get_logger().info(f"Viewing {topic}; press q or Esc to quit.")

    def image_callback(self, message: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not convert camera image: {exc}")
            return

        self.frames_received += 1

        if self.scale != 1.0:
            width = int(frame.shape[1] * self.scale)
            height = int(frame.shape[0] * self.scale)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/sim/rail_franka1/cam/wrist/color/image_raw",
        help="ROS Image topic to display",
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
    viewer = WristCameraViewer(args.topic, args.window, args.scale)
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
