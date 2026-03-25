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
        super().__init__('interactive_joint_control_loop')
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

    def interpolate_trajectory(self, start, end, duration, rate_hz):
        steps = int(duration * rate_hz)
        for t in range(steps + 1):
            alpha = float(t) / steps
            yield (1 - alpha) * np.array(start) + alpha * np.array(end)

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
        print("\n🎮 Interactive Franka Joint Controller")
        print("💡 Use shortcuts: j1–j7 (arm), f1–f2 (fingers), type 'done' or 'exit' to finish.\n")
        joint_targets = {}

        while True:
            try:
                inp = input("🧩 Enter joint and position (e.g. j2 -1.0): ").strip()
            except EOFError:
                return None, None

            if inp.lower() in ['done', 'exit']:
                break

            parts = inp.split()
            if len(parts) != 2:
                print("❗ Format: <joint> <position>, e.g. j2 -1.0")
                continue

            short_name, pos_str = parts
            if short_name not in alias_map:
                print(f"❗ Unknown joint alias: '{short_name}'")
                continue

            try:
                pos_val = float(pos_str)
            except ValueError:
                print(f"❗ Invalid number: {pos_str}")
                continue

            joint_targets[alias_map[short_name]] = pos_val

        if not joint_targets:
            print("❗ No valid joints entered.")
            return None, None

        duration = input("⏱️  Enter duration in seconds [default: 3.0]: ").strip()
        try:
            duration = float(duration) if duration else 3.0
        except:
            duration = 3.0

        print("\n📋 Command Summary:")
        for joint, val in joint_targets.items():
            print(f"  {joint}: {val}")
        print(f"  Duration: {duration:.2f} sec\n")

        confirm = input("✅ Proceed? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled.")
            return None, None

        return joint_targets, duration

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

                with self.lock:
                    target_positions = self.current_joint_positions[:]
                    for joint, val in target_subset.items():
                        idx = joint_names.index(joint)
                        target_positions[idx] = val
                    start_positions = self.current_joint_positions[:]

                for pos in self.interpolate_trajectory(start_positions, target_positions, duration, 50):
                    msg = JointState()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.name = joint_names
                    msg.position = pos.tolist()
                    self.publisher.publish(msg)
                    rclpy.spin_once(self, timeout_sec=0.001)
                    time.sleep(1.0 / 50)

                with self.lock:
                    self.current_joint_positions = target_positions

                print("✅ Movement complete.\n")

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

