#!/usr/bin/env python3
"""Publish the rail-zero global frame without depending on Isaac's world TF.

The node consumes the rail encoder value already exposed as ``rail_j1`` in a
``sensor_msgs/JointState`` message.  It makes ``global_origin`` the TF root
and publishes the moving robot base below it.  Thus rail_j1 == 0 is always the
global origin, independent of when this node or the orchestrator starts.
"""

from __future__ import annotations

import math

import rclpy
import ros_logger
ros_logger.setup_ros_logging()
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

def rail_zero_base_pose(
    rail_position: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return global_origin -> panda_link0 translation and XYZW quaternion.

    ``rail_j1`` is the rail-aligned global X coordinate in metres.  The base
    axes are rotated 180 degrees about Z relative to those rail axes.
    """
    if not math.isfinite(rail_position):
        raise ValueError("rail_j1 must be finite")
    return (float(rail_position), 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)



def rail_transform(parent_frame: str, base_frame: str, rail_position: float) -> TransformStamped:
    """Build global_origin -> panda_link0 for a rail reading in metres."""
    translation, quaternion_xyzw = rail_zero_base_pose(rail_position)

    transform = TransformStamped()
    transform.header.frame_id = parent_frame
    transform.child_frame_id = base_frame
    transform.transform.translation.x = translation[0]
    transform.transform.translation.y = translation[1]
    transform.transform.translation.z = translation[2]
    transform.transform.rotation.x = quaternion_xyzw[0]
    transform.transform.rotation.y = quaternion_xyzw[1]
    transform.transform.rotation.z = quaternion_xyzw[2]
    transform.transform.rotation.w = quaternion_xyzw[3]
    return transform


class RailGlobal(Node):
    """Publish a rail-derived base transform after checking TF parent safety."""

    def __init__(self) -> None:
        super().__init__("rail_global")
        self.declare_parameter("joint_states_topic", "/sim/rail_franka1/joint_states")
        self.declare_parameter("rail_joint", "rail_j1")
        self.declare_parameter("global_frame", "global_origin")
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("parent_check_sec", 2.0)

        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.rail_joint = str(self.get_parameter("rail_joint").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.parent_check_sec = float(self.get_parameter("parent_check_sec").value)
        if self.global_frame == self.base_frame:
            raise ValueError("global_frame and base_frame must be different")

        self._other_parent: str | None = None
        self._ready_to_publish = False
        self._failed = False
        self._parent_check_started = False
        self._parent_check_timer = None
        self._latest_rail_position: float | None = None
        self._latest_rail_stamp = None
        self._broadcaster = TransformBroadcaster(self)
        self._joint_subscription = self.create_subscription(
            JointState, self.joint_states_topic, self._joint_callback, 20
        )
        self.create_subscription(TFMessage, "/tf", self._observe_tf, qos_profile_sensor_data)
        static_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(TFMessage, "/tf_static", self._observe_tf, static_qos)
        self.get_logger().info(
            f"Waiting for {self.rail_joint!r} telemetry before checking TF parents for "
            f"{self.base_frame!r}."
        )

    def _observe_tf(self, message: TFMessage) -> None:
        if self._ready_to_publish or self._failed:
            return
        for transform in message.transforms:
            if (
                transform.child_frame_id == self.base_frame
                and transform.header.frame_id != self.global_frame
            ):
                self._other_parent = transform.header.frame_id
                return

    def _finish_parent_check(self) -> None:
        if self._parent_check_timer is not None:
            self._parent_check_timer.cancel()
        if self._other_parent is not None:
            self._failed = True
            self.get_logger().fatal(
                f"Refusing to publish {self.global_frame!r} -> {self.base_frame!r}: "
                f"{self.base_frame!r} is already a child of {self._other_parent!r}."
            )
            return
        self._ready_to_publish = True
        self.get_logger().info(
            f"Publishing rail-zero global TF {self.global_frame!r} -> {self.base_frame!r} "
            f"from {self.joint_states_topic!r}."
        )
        if self._latest_rail_position is not None:
            self._publish_transform(self._latest_rail_position, self._latest_rail_stamp)

    def _joint_callback(self, message: JointState) -> None:
        if self._failed:
            return
        try:
            index = message.name.index(self.rail_joint)
            rail_position = float(message.position[index])
            rail_transform(self.global_frame, self.base_frame, rail_position)
        except (ValueError, IndexError) as exc:
            self.get_logger().warning(f"Ignoring invalid {self.rail_joint!r} telemetry: {exc}")
            return

        self._latest_rail_position = rail_position
        self._latest_rail_stamp = message.header.stamp
        if not self._parent_check_started:
            self._parent_check_started = True
            self._parent_check_timer = self.create_timer(
                max(0.01, self.parent_check_sec), self._finish_parent_check
            )
            self.get_logger().info(
                f"Received {self.rail_joint!r}; checking TF parents for "
                f"{self.parent_check_sec:g}s before publishing."
            )
            return
        if self._ready_to_publish:
            self._publish_transform(rail_position, message.header.stamp)

    def _publish_transform(self, rail_position: float, stamp) -> None:
        transform = rail_transform(self.global_frame, self.base_frame, rail_position)
        transform.header.stamp = stamp
        if transform.header.stamp.sec == 0 and transform.header.stamp.nanosec == 0:
            transform.header.stamp = self.get_clock().now().to_msg()
        self._broadcaster.sendTransform(transform)


def main() -> None:
    rclpy.init()
    node = RailGlobal()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
