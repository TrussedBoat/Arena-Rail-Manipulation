#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class LocalImagePublisher(Node):
    def __init__(self):
        super().__init__('local_image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0, self.timer_callback) # Publish every second
        self.bridge = CvBridge()
        
        # Path to your target screenshot
        self.image_path = "/home/homerobotics/Pictures/Screenshots/exp4-2.png"

    def timer_callback(self):
        cv_image = cv2.imread(self.image_path)
        if cv_image is not None:
            # Convert OpenCV image to ROS 2 Image message
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.publisher_.publish(ros_image)
            self.get_logger().info('Publishing screenshot frame...')
        else:
            self.get_logger().error(f'Failed to load image from {self.image_path}')

def main(args=None):
    rclpy.init(args=args)
    node = LocalImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()