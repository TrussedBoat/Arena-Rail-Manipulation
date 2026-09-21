#!/usr/bin/env python3
"""Display and record the simulated Panda wrist camera in an OpenCV window."""

import argparse
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class WristCameraRecorder(Node):
    def __init__(self, color_topic: str, window: str, scale: float = 1.0, out_file: str = "wrist_recording.mp4") -> None:
        super().__init__("wrist_camera_recorder")
        self.window = window
        self.scale = scale
        self.out_file = out_file
        self.bridge = CvBridge()
        
        self.is_recording = False
        self.video_writer = None
        
        self.color_sub = self.create_subscription(
            Image,
            color_topic,
            self.color_callback,
            10,
        )
        self.get_logger().info(f"Viewing color: {color_topic}")
        self.get_logger().info("Controls: 'r' to toggle recording, 'q' or Esc to quit.")

    def color_callback(self, message: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not convert color image: {exc}")
            return
            
        if self.scale != 1.0:
            width = int(frame.shape[1] * self.scale)
            height = int(frame.shape[0] * self.scale)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Handle recording logic
        if self.is_recording:
            if self.video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.out_file, fourcc, 30.0, (w, h))
                self.get_logger().info(f"Started recording to {self.out_file}...")
            
            self.video_writer.write(frame)
            
            # Add a red recording dot to the UI
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)

        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            if self.video_writer is not None:
                self.video_writer.release()
                self.get_logger().info(f"Saved recording to {self.out_file}")
            rclpy.shutdown()
        elif key == ord('r'):
            self.is_recording = not self.is_recording
            if not self.is_recording and self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self.get_logger().info(f"Stopped recording. Saved to {self.out_file}")
                # Generate new filename for subsequent recordings
                self.out_file = f"wrist_recording_{int(time.time())}.mp4"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/sim/rail_franka1/cam/wrist/color/image_raw",
        help="ROS Color Image topic to display",
    )
    parser.add_argument("--window", default="Panda wrist camera (Recorder)")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for the display window",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="wrist_recording.mp4",
        help="Output mp4 filename",
    )
    args = parser.parse_args()

    rclpy.init()
    node = WristCameraRecorder(
        color_topic=args.topic,
        window=args.window,
        scale=args.scale,
        out_file=args.out
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.video_writer is not None:
            node.video_writer.release()
            node.get_logger().info(f"Saved recording to {node.out_file}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0

if __name__ == "__main__":
    exit(main())
