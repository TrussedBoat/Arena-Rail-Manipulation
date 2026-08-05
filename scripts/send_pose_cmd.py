#!/usr/bin/env python3
"""Send one Cartesian EEF target to the Arena controller without the VLM.

The controller subscribes to geometry_msgs/Pose on /pose_cmd and interprets it
in the panda_link0 frame.  This utility accepts XYZ in metres and fixed-axis
XYZ roll/pitch/yaw in radians, matching tools.move_eef_to_pose().
"""

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Bool


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert fixed-axis XYZ RPY (radians) to an xyzw quaternion."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class PoseCommand(Node):
    def __init__(self) -> None:
        super().__init__("pose_cmd_debugger")
        self.publisher = self.create_publisher(Pose, "/pose_cmd", 10)
        self.controller_ready = False
        self.command_complete = False
        self.create_subscription(Bool, "/controller_ready", self._ready_callback, 10)
        self.create_subscription(Bool, "/controller_state", self._state_callback, 10)

    def _ready_callback(self, message: Bool) -> None:
        self.controller_ready = message.data

    def _state_callback(self, message: Bool) -> None:
        if message.data:
            self.command_complete = True


def spin_until(node: PoseCommand, predicate, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return True
    return predicate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, help="EEF x in panda_link0, metres")
    parser.add_argument("y", type=float, help="EEF y in panda_link0, metres")
    parser.add_argument("z", type=float, help="EEF z in panda_link0, metres")
    parser.add_argument("roll", type=float, help="roll, radians")
    parser.add_argument("pitch", type=float, help="pitch, radians")
    parser.add_argument("yaw", type=float, help="yaw, radians")
    parser.add_argument("--ready-timeout", type=float, default=5.0)
    parser.add_argument("--completion-timeout", type=float, default=30.0)
    parser.add_argument("--no-wait", action="store_true", help="publish and exit without waiting for completion")
    args = parser.parse_args()

    values = (args.x, args.y, args.z, args.roll, args.pitch, args.yaw)
    if not all(math.isfinite(value) for value in values):
        parser.error("all pose values must be finite")

    rclpy.init()
    node = PoseCommand()
    try:
        ready = spin_until(
            node,
            lambda: node.controller_ready or node.publisher.get_subscription_count() > 0,
            args.ready_timeout,
        )
        if not ready:
            node.get_logger().error("No Cartesian controller subscriber found on /pose_cmd")
            return 2

        quaternion = quaternion_from_rpy(args.roll, args.pitch, args.yaw)
        message = Pose()
        message.position.x, message.position.y, message.position.z = args.x, args.y, args.z
        message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = quaternion
        node.command_complete = False
        node.publisher.publish(message)
        node.get_logger().info(
            "Published /pose_cmd in panda_link0: "
            f"xyz=({args.x:.4f}, {args.y:.4f}, {args.z:.4f}), "
            f"rpy=({args.roll:.4f}, {args.pitch:.4f}, {args.yaw:.4f})"
        )
        if args.no_wait:
            return 0
        if not spin_until(node, lambda: node.command_complete, args.completion_timeout):
            node.get_logger().error("Timed out waiting for /controller_state=true")
            return 3
        node.get_logger().info("Controller reported command completion.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
