#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class MultiCameraStreamer(Node):
    def __init__(self, depth=True):
        super().__init__('multi_camera_video_streamer')
        self.bridge = CvBridge()
        self.depth = depth

        # Store latest images
        self.latest_images = {
            "tripod_1": None,
            "tripod_2": None,
            "wrist": None
        }

        # Define topics
        self.image_topics = {
            "tripod_1": "/tripod_1/rgb",
            "tripod_2": "/tripod_2/rgb",
            "wrist": "/wrist/rgb"
        }

        # Subscribe to all topics
        self.subscribers = []
        for cam_name, topic in self.image_topics.items():
            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, name=cam_name: self.image_callback(msg, name),
                10
            )
            self.subscribers.append(sub)

        # Timer for streaming at ~10Hz
        self.timer = self.create_timer(0.1, self.stream_video)
        self.get_logger().info("Multi-camera video streamer started.")

    def image_callback(self, msg, source_name):
        try:
            # RGB image
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_images[source_name] = image
        except Exception as e:
            self.get_logger().error(f"Error processing {source_name} image: {e}")

    def stream_video(self):
        # Create list of all images
        images = []
        for cam in self.image_topics.keys():
            img = self.latest_images[cam]
            if img is None:
                # Placeholder if no image yet
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(img, f"Waiting for {cam}", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                img = cv2.resize(img, (640, 360))  # Standardize size
            images.append(img)

        # Arrange in a 3x1 grid (since depth images removed)
        grid = np.vstack(images)

        cv2.imshow("Multi-Camera Stream", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quit requested")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraStreamer(depth=False)  # RGB only
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

