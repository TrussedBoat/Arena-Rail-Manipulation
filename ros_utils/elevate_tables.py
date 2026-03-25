#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class TableElevationController(Node):
    def __init__(self, topic_prefix, elevation_cm):
        super().__init__('table_elevation_controller')

        topic_name = f'{topic_prefix}/joint_command'
        self.publisher_ = self.create_publisher(JointState, topic_name, 10)

        # Clamp elevation between 0 and 60 cm
        elevation_cm = max(0.0, min(60.0, elevation_cm))
        self.elevation_m = elevation_cm / 100.0  # convert to meters

        self.get_logger().info(
            f"Holding elevation at {elevation_cm:.1f} cm ({self.elevation_m:.3f} m) "
            f"on topic '{topic_name}'... Press Ctrl+C to stop."
        )

        # Create the JointState message (static parts)
        self.joint_msg = JointState()
        self.joint_msg.name = ['j1', 'j2', 'j3', 'j4']
        self.joint_msg.velocity = [0.14] * 4
        self.joint_msg.effort = [0.0] * 4

        # Timer to send command at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        self.joint_msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_msg.position = [self.elevation_m] * 4
        self.publisher_.publish(self.joint_msg)


def main():
    rclpy.init()

    topic_prefix = input("Enter topic prefix (example: /t0): ").strip()
    try:
        elevation_cm = float(input("Enter elevation in cm (0–60): ").strip())
    except ValueError:
        print("Invalid elevation value. Must be a number.")
        rclpy.shutdown()
        return

    node = TableElevationController(topic_prefix, elevation_cm)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

