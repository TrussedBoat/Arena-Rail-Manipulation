#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import numpy as np
import ros2_numpy as rnp

class DummyPublisher(Node):

    def __init__(self):
        super().__init__('dummy_publisher')
        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 1)
        self.tri1_pub = self.create_publisher(Image, '/tripod_1/rgb', 1)
        self.tri2_pub = self.create_publisher(Image, '/tripod_2/rgb', 1)
        self.wrist_pub = self.create_publisher(Image, '/wrist/rgb', 1)
        self.grip_pub = self.create_publisher(Float32MultiArray, '/gripper_position', 1)
        self.timer = self.create_timer(0.5, self.publish_dummy)

    def publish_dummy(self):
        # 1. joint_states topic
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [f'panda_joint{i}' for i in range(1,8)]
        js.position = [float(i) / 10.0 for i in range(7)]
        self.joint_pub.publish(js)

        # 2–4. rgb image topics
        for pub in (self.tri1_pub, self.tri2_pub, self.wrist_pub):
            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img_msg = rnp.msgify(Image, arr, encoding='rgb8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            pub.publish(img_msg)

        # 5. gripper_position topic
        grip = Float32MultiArray()
        grip.data = [np.random.random()]
        self.grip_pub.publish(grip)

        self.get_logger().info('Published dummy data')

def main():
    rclpy.init()
    node = DummyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

