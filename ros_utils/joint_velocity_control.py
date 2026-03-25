#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import threading
import time

joint_names = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
    'panda_joint5', 'panda_joint6', 'panda_joint7',
    'panda_finger_joint1', 'panda_finger_joint2'
]

alias_map = {
    'j1': 'panda_joint1', 'j2': 'panda_joint2', 'j3': 'panda_joint3',
    'j4': 'panda_joint4', 'j5': 'panda_joint5', 'j6': 'panda_joint6',
    'j7': 'panda_joint7', 'f1': 'panda_finger_joint1', 'f2': 'panda_finger_joint2'
}

class JointControlNode(Node):
    def __init__(self):
        super().__init__('joint_velocity_control')
        self.current_joint_positions = None
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.publisher = self.create_publisher(JointState, '/joint_command', 10)
        self.lock = threading.Lock()

    def joint_state_callback(self, msg):
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        try:
            with self.lock:
                self.current_joint_positions = [msg.position[name_to_index[name]] for name in joint_names]
        except KeyError:
            self.get_logger().warn('Received joint_states missing required joints.')

    def wait_for_joint_states(self, timeout_sec=5.0):
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            with self.lock:
                if self.current_joint_positions is not None:
                    return True
            if time.time() - start_time > timeout_sec:
                return False
        return False

    def prompt_user(self):
        print("\n🎮 Interactive Franka Joint Velocity Controller")
        print("💡 Use shortcuts: j1–j7 (arm), f1–f2 (fingers), type 'done' or 'exit' to finish.\n")
        joint_velocities = {}

        while True:
            try:
                inp = input("🧩 Enter joint and velocity (e.g. j2 -1.0): ").strip()
            except EOFError:
                return None, None

            if inp.lower() in ['done', 'exit']:
                break

            parts = inp.split()
            if len(parts) != 2:
                print("❗ Format: <joint> <velocity>, e.g. j2 -1.0")
                continue

            short_name, vel_str = parts
            if short_name not in alias_map:
                print(f"❗ Unknown joint alias: '{short_name}'")
                continue

            try:
                vel_val = float(vel_str)
            except ValueError:
                print(f"❗ Invalid number: {vel_str}")
                continue

            joint_velocities[alias_map[short_name]] = vel_val

        if not joint_velocities:
            print("❗ No valid joints entered.")
            return None, None

        duration = input("⏱️  Enter duration in seconds [default: 3.0]: ").strip()
        try:
            duration = float(duration) if duration else 3.0
        except:
            duration = 3.0

        print("\n📋 Command Summary:")
        for joint, val in joint_velocities.items():
            print(f"  {joint}: {val}")
        print(f"  Duration: {duration:.2f} sec\n")

        confirm = input("✅ Proceed? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled.")
            return None, None

        return joint_velocities, duration

    def run_loop(self):
        if not self.wait_for_joint_states():
            self.get_logger().error("❌ Timeout: Couldn't receive /joint_states.")
            return

        print("✅ Joint states received. Ready to control!")

        try:
            while rclpy.ok():
                target_subset, duration = self.prompt_user()
                if target_subset is None:
                    continue

                velocity_command = JointState()
                velocity_command.header.stamp = self.get_clock().now().to_msg()
                velocity_command.name = joint_names
                velocity_command.velocity = [0.0] * len(joint_names)

                for joint, vel in target_subset.items():
                    idx = joint_names.index(joint)
                    velocity_command.velocity[idx] = vel

                print("🚀 Sending velocity command...")
                start_time = time.time()

                while time.time() - start_time < duration and rclpy.ok():
                    velocity_command.header.stamp = self.get_clock().now().to_msg()
                    self.publisher.publish(velocity_command)
                    rclpy.spin_once(self, timeout_sec=0.001)
                    time.sleep(1.0 / 50)  # 50 Hz

                # Send zero velocity to stop
                stop_command = JointState()
                stop_command.header.stamp = self.get_clock().now().to_msg()
                stop_command.name = joint_names
                stop_command.velocity = [0.0] * len(joint_names)

                print("🛑 Stopping the robot.")
                for _ in range(5):  # ensure stop command is received
                    self.publisher.publish(stop_command)
                    time.sleep(0.05)

                print("✅ Velocity command complete.\n")

        except KeyboardInterrupt:
            print("\n🚪 Exiting gracefully.")

def main():
    rclpy.init()
    node = JointControlNode()
    node.run_loop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

