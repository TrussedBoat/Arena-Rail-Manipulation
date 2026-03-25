#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from math import radians

class InteractivePandaController(Node):
    def __init__(self):
        super().__init__('interactive_panda_controller')
        self.publisher = self.create_publisher(JointState, '/joint_command', 10)
        self.timer = self.create_timer(0.5, self.loop)
        self.ready = True

        self.panda_joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        self.gripper_names = ['panda_finger_joint1', 'panda_finger_joint2']
        self.rail_joint_name = 'franka1_prismatic'

        self.default_panda_angles_deg = [0.0, -30.0, 0.0, -120.0, 0.0, 120.0, 45.0]
        self.default_gripper = 0.04
        self.default_rail = 0.0

        print("🚀 Ready to control Panda arm and prismatic rail in Isaac Sim.")
        print("Press Ctrl+C to exit.\n")

    def loop(self):
        if not self.ready:
            return
        self.ready = False

        try:
            choice = input("🔀 What do you want to move? (panda / prismatic): ").strip().lower()

            if choice == "prismatic":
                pos_str = input(f"📏 Rail position in meters (default={self.default_rail}): ").strip()
                pos = float(pos_str) if pos_str else self.default_rail
                self.send_prismatic_command(pos)

            elif choice == "panda":
                positions = []
                print("🤖 Enter Panda joint angles in degrees:")
                for i, name in enumerate(self.panda_joint_names):
                    default = self.default_panda_angles_deg[i]
                    angle_str = input(f"  {name} (default={default}°): ").strip()
                    angle = float(angle_str) if angle_str else default
                    positions.append(radians(angle))

                grip_str = input(f"🤏 Gripper opening in meters (0 to 0.04, default={self.default_gripper}): ").strip()
                gripper_pos = float(grip_str) if grip_str else self.default_gripper

                self.send_panda_command(positions, gripper_pos)
            else:
                print("❌ Invalid choice. Type 'panda' or 'prismatic'.")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            print("")
            self.ready = True  # allow next iteration

    def send_prismatic_command(self, position):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self.rail_joint_name]
        msg.position = [position]
        msg.velocity = [0.0]
        msg.effort = [0.0]
        self.publisher.publish(msg)
        self.get_logger().info(f"✅ Moved rail: {self.rail_joint_name} → {position:.3f} m")

    def send_panda_command(self, joint_angles_rad, gripper_pos):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.panda_joint_names + self.gripper_names
        msg.position = joint_angles_rad + [gripper_pos, gripper_pos]
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        self.publisher.publish(msg)
        self.get_logger().info("✅ Moved Panda joints + gripper.")

def main():
    rclpy.init()
    node = InteractivePandaController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n👋 Exiting on user request.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

