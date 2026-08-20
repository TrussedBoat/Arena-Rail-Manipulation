#!/usr/bin/env python3
"""Central command/ownership bridge between ROS controllers and Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64

ORCHESTRATOR_SRC = Path(__file__).resolve().parents[1] / "agent_orchestrator" / "src"
if str(ORCHESTRATOR_SRC) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_SRC))

from config import load_runtime_config
from interface.action import DirectJointCommand
from interface.srv import AcquireArmLease, EmergencyArmHold, ReleaseArmLease


PANDA_ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
RAIL_JOINT = "rail_j1"
GRIPPER_JOINT = "panda_finger_joint1"
DIRECT_TIMEOUT_SEC = 15.0
DEFAULT_RAIL_COMMAND_TIMEOUT_SEC = 30.0
GRIPPER_SETTLE_SEC = 4.0
STABLE_SAMPLES_REQUIRED = 5
SIM_COMMAND_REPUBLISH_HZ = 100.0
VELOCITY_SETTLE_THRESHOLD = 0.25
GRIPPER_POSITION_TOLERANCE_M = 0.005
RAIL_POSITION_TOLERANCE_M = 0.02


@dataclass
class DirectGoal:
    resource: str
    goal_handle: object
    joint_names: tuple[str, ...]
    targets: dict[str, float]
    velocities: dict[str, float]
    completion_policy: str
    started_at: float
    feedback_sequence_at_start: int
    stable_samples: int = 0
    state: str = "executing"
    reason: str = ""
    done: threading.Event = field(default_factory=threading.Event)


class RailUnifiedBridge(Node):
    """Authoritative lease manager and persistent Isaac command publisher."""

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


    def __init__(self) -> None:
        super().__init__("rail_unified_bridge_node")
        self._callbacks = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self.direct_target_position_tolerance = load_runtime_config().search.wrist_joint_tolerance
        self.declare_parameter("rail_damping_time_constant_sec", 0.18)
        self.declare_parameter("rail_damping_max_speed_mps", 0.35)
        self.declare_parameter("rail_command_timeout_sec", DEFAULT_RAIL_COMMAND_TIMEOUT_SEC)
        self.rail_damping_time_constant_sec = float(
            self.get_parameter("rail_damping_time_constant_sec").value
        )
        self.rail_damping_max_speed_mps = float(
            self.get_parameter("rail_damping_max_speed_mps").value
        )
        self.rail_command_timeout_sec = float(
            self.get_parameter("rail_command_timeout_sec").value
        )
        if self.rail_damping_time_constant_sec <= 0.0:
            raise ValueError("rail_damping_time_constant_sec must be positive")
        if self.rail_damping_max_speed_mps <= 0.0:
            raise ValueError("rail_damping_max_speed_mps must be positive")
        if self.rail_command_timeout_sec <= 0.0:
            raise ValueError("rail_command_timeout_sec must be positive")

        self._actual_positions: dict[str, float] = {}
        self._actual_velocities: dict[str, float] = {}
        self._feedback_sequence = 0
        self._direct_goals: dict[str, DirectGoal] = {}
        self._arm_lease: dict[str, object] | None = None
        self._filtered_rail_target: float | None = None
        self._last_rail_filter_update = self._now()

        self.sys1_cmd_cache = {name: 0.0 for name in (*PANDA_ARM_JOINTS, RAIL_JOINT, GRIPPER_JOINT)}
        self.sys1_cmd_vel_cache = {name: 0.0 for name in self.sys1_cmd_cache}
        self.sys1_cmd_received: set[str] = set()
        self.sys2_cmd_cache: dict[str, float] = {}
        self.sys2_cmd_vel_cache: dict[str, float] = {}
        self.sys2_cmd_received: set[str] = set()

        transient_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_sys1_cmd = self.create_publisher(JointState, "/sim/rail_franka1/joint_command", 10)
        self.pub_sys2_cmd = self.create_publisher(JointState, "/sim/rail_franka2/joint_command", 10)
        self.pub_rail1_state = self.create_publisher(JointState, "/sim/rail1/joint_states", 10)
        self.pub_franka1_state = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_gripper_state = self.create_publisher(Float64, "/gripper_state", 10)
        self.pub_rail2_state = self.create_publisher(JointState, "/sim/rail2/joint_states", 10)
        self.pub_franka2_state = self.create_publisher(JointState, "/sim/franka2_rail/joint_states", 10)
        self.direct_control_state_pub = self.create_publisher(
            Bool, "/panda/direct_joint_control_active", transient_qos
        )

        self.create_subscription(JointState, "/sim/rail_franka1/joint_states", self._state1_callback, 20, callback_group=self._callbacks)
        self.create_subscription(JointState, "/sim/rail_franka2/joint_states", self._state2_callback, 20, callback_group=self._callbacks)
        self.create_subscription(JointState, "/cartesian/joint_command", self._cartesian_joint_callback, 20, callback_group=self._callbacks)
        self.create_subscription(JointState, "/rrt/hold_command", self._safety_hold_callback, 10, callback_group=self._callbacks)
        self.create_subscription(JointState, "/direct_joint_command", self._legacy_joint_callback, 10, callback_group=self._callbacks)
        self.create_subscription(Float64, "/gripper/command", self._legacy_gripper_callback, 10, callback_group=self._callbacks)
        self.create_subscription(Pose, "/rrt/pose_command", self._legacy_pose_callback, 10, callback_group=self._callbacks)
        self.create_subscription(Pose, "/panda/cartesian_pose_command", self._legacy_pose_callback, 10, callback_group=self._callbacks)
        self.create_subscription(JointState, "/sim/rail2/joint_command", self._sys2_command_callback, 10, callback_group=self._callbacks)
        self.create_subscription(JointState, "/sim/franka2_rail/joint_command", self._sys2_command_callback, 10, callback_group=self._callbacks)

        self._direct_action = ActionServer(
            self, DirectJointCommand, "/bridge/direct_joint_command",
            execute_callback=self._execute_direct_goal,
            goal_callback=self._direct_goal_callback,
            cancel_callback=self._direct_cancel_callback,
            callback_group=self._callbacks,
        )
        self.create_service(AcquireArmLease, "/bridge/acquire_arm_lease", self._acquire_lease_callback, callback_group=self._callbacks)
        self.create_service(ReleaseArmLease, "/bridge/release_arm_lease", self._release_lease_callback, callback_group=self._callbacks)
        self.create_service(EmergencyArmHold, "/bridge/emergency_arm_hold", self._emergency_hold_callback, callback_group=self._callbacks)
        self.create_timer(1.0 / SIM_COMMAND_REPUBLISH_HZ, self._periodic_update, callback_group=self._callbacks)
        self._publish_direct_control_state()
        self.get_logger().info(
            "Unified bridge ready: direct actions + atomic arm leases + 100 Hz simulator setpoints "
            f"(rail damping tau={self.rail_damping_time_constant_sec:.2f}s, "
            f"max_speed={self.rail_damping_max_speed_mps:.2f}m/s, "
            f"rail timeout={self.rail_command_timeout_sec:.1f}s)"
        )

    @staticmethod
    def _is_arm_joint(name: str) -> bool:
        return name in PANDA_ARM_JOINTS

    def _arm_owner(self) -> str:
        return "none" if self._arm_lease is None else str(self._arm_lease["owner"])

    def _publish_direct_control_state(self) -> None:
        self.direct_control_state_pub.publish(Bool(data=self._arm_owner() == "direct_arm"))

    def _lease_details_locked(self) -> tuple[str, str, float, float]:
        if self._arm_lease is None:
            return "none", "", 0.0, 0.0
        age = max(0.0, self._now() - float(self._arm_lease["started_at"]))
        remaining = max(0.0, DIRECT_TIMEOUT_SEC - age) if self._arm_lease["owner"] == "direct_arm" else 0.0
        return str(self._arm_lease["owner"]), str(self._arm_lease["execution_id"]), age, remaining

    def _hold_arm_locked(self) -> None:
        for name in PANDA_ARM_JOINTS:
            if name in self._actual_positions:
                self.sys1_cmd_cache[name] = self._actual_positions[name]
                self.sys1_cmd_vel_cache[name] = 0.0
                self.sys1_cmd_received.add(name)

    def _release_arm_locked(self, lease_id: str | None = None) -> bool:
        if self._arm_lease is None or (lease_id is not None and self._arm_lease["lease_id"] != lease_id):
            return False
        self._hold_arm_locked()
        self._arm_lease = None
        self._publish_direct_control_state()
        return True

    def _acquire_lease_callback(self, request, response):
        with self._lock:
            if self._arm_lease is None:
                lease_id = uuid.uuid4().hex
                self._arm_lease = {
                    "owner": str(request.owner), "execution_id": str(request.execution_id),
                    "lease_id": lease_id, "started_at": self._now(),
                }
                self._publish_direct_control_state()
                response.granted, response.lease_id, response.reason = True, lease_id, "granted"
                response.active_owner, response.active_execution_id = str(request.owner), str(request.execution_id)
                response.owner_age_sec, response.owner_remaining_sec = 0.0, 0.0
                return response
            owner, execution_id, age, remaining = self._lease_details_locked()
            response.granted, response.lease_id = False, ""
            response.reason = f"blocked_by_{owner}"
            response.active_owner, response.active_execution_id = owner, execution_id
            response.owner_age_sec, response.owner_remaining_sec = age, remaining
            return response

    def _release_lease_callback(self, request, response):
        with self._lock:
            response.released = self._release_arm_locked(str(request.lease_id))
            response.reason = "released" if response.released else "lease_not_active_or_mismatched"
            return response

    def _emergency_hold_callback(self, request, response):
        with self._lock:
            if self._arm_lease is not None and self._arm_lease["lease_id"] != str(request.lease_id):
                response.held, response.reason = False, "lease_not_active_or_mismatched"
                return response
            self._hold_arm_locked()
            self._arm_lease = None
            self._publish_direct_control_state()
            response.held, response.reason = True, str(request.reason) or "emergency_hold"
            return response

    def _direct_goal_callback(self, request):
        resource, names = str(request.resource).strip().lower(), tuple(str(name) for name in request.joint_names)
        if resource not in {"arm", "rail", "gripper"} or not names or len(names) != len(request.positions):
            return GoalResponse.REJECT
        valid = (
            resource == "arm" and all(self._is_arm_joint(name) for name in names)
            or resource == "rail" and names == (RAIL_JOINT,)
            or resource == "gripper" and names == (GRIPPER_JOINT,)
        )
        if not valid or not all(np.isfinite(float(value)) for value in request.positions):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _direct_cancel_callback(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _finish_direct_locked(self, record: DirectGoal, state: str, reason: str, *, hold: bool = True) -> None:
        if record.done.is_set():
            return
        record.state, record.reason = state, reason
        if self._direct_goals.get(record.resource) is record:
            self._direct_goals.pop(record.resource, None)
        if hold and state != "superseded":
            for name in record.joint_names:
                if name in self._actual_positions:
                    self.sys1_cmd_cache[name] = self._actual_positions[name]
                    self.sys1_cmd_vel_cache[name] = 0.0
                    self.sys1_cmd_received.add(name)
        if record.resource == "arm" and state != "superseded" and self._arm_owner() == "direct_arm":
            self._release_arm_locked(str(self._arm_lease["lease_id"]))
        record.done.set()

    def _install_direct_goal_locked(self, goal_handle) -> DirectGoal | None:
        request = goal_handle.request
        resource = str(request.resource).strip().lower()
        if resource == "arm" and self._arm_lease is not None and self._arm_owner() != "direct_arm":
            return None
        old = self._direct_goals.get(resource)
        if old is not None:
            self._finish_direct_locked(old, "superseded", "replaced by newer same-resource goal", hold=False)
        if resource == "arm" and self._arm_lease is None:
            self._arm_lease = {
                "owner": "direct_arm", "execution_id": uuid.uuid4().hex,
                "lease_id": uuid.uuid4().hex, "started_at": self._now(),
            }
            self._publish_direct_control_state()
        names = tuple(str(name) for name in request.joint_names)
        record = DirectGoal(
            resource=resource, goal_handle=goal_handle, joint_names=names,
            targets={name: float(value) for name, value in zip(names, request.positions)},
            velocities={name: float(request.velocities[index]) for index, name in enumerate(names)
                        if index < len(request.velocities) and np.isfinite(float(request.velocities[index]))},
            completion_policy=str(request.completion_policy or "target_tolerance"),
            started_at=self._now(), feedback_sequence_at_start=self._feedback_sequence,
        )
        self._direct_goals[resource] = record
        for name, target in record.targets.items():
            self.sys1_cmd_cache[name] = target
            self.sys1_cmd_vel_cache[name] = record.velocities.get(name, 0.0)
            self.sys1_cmd_received.add(name)
        return record

    def _direct_result(self, record: DirectGoal):
        result = DirectJointCommand.Result()
        result.success, result.state, result.reason = record.state == "reached", record.state, record.reason
        result.joint_names = list(record.joint_names)
        with self._lock:
            result.final_positions = [float(self._actual_positions.get(name, self.sys1_cmd_cache.get(name, 0.0))) for name in record.joint_names]
        return result

    def _feedback_direct(self, record: DirectGoal) -> None:
        feedback = DirectJointCommand.Feedback()
        feedback.state, feedback.reason = record.state, record.reason or "executing"
        feedback.elapsed_sec = float(self._now() - record.started_at)
        feedback.joint_names = list(record.joint_names)
        with self._lock:
            feedback.measured_positions = [float(self._actual_positions.get(name, 0.0)) for name in record.joint_names]
        record.goal_handle.publish_feedback(feedback)

    def _execute_direct_goal(self, goal_handle):
        with self._lock:
            record = self._install_direct_goal_locked(goal_handle)
            if record is None:
                owner, execution_id, age, remaining = self._lease_details_locked()
                record = DirectGoal("arm", goal_handle, tuple(), {}, {}, "target_tolerance", self._now(), self._feedback_sequence,
                                    state="blocked_by_cartesian", reason=f"arm owned by {owner} execution={execution_id} age={age:.2f}s remaining={remaining:.2f}s")
                record.done.set()
        last_feedback = 0.0
        while not record.done.wait(timeout=0.05):
            if goal_handle.is_cancel_requested:
                with self._lock:
                    self._finish_direct_locked(record, "cancelled", "action cancellation requested")
                break
            if self._now() - last_feedback >= 0.25:
                self._feedback_direct(record)
                last_feedback = self._now()
        result = self._direct_result(record)
        if record.state == "reached": goal_handle.succeed()
        elif record.state == "cancelled": goal_handle.canceled()
        else: goal_handle.abort()
        return result

    def _direct_goal_reached_locked(self, record: DirectGoal) -> tuple[bool, str]:
        if self._feedback_sequence <= record.feedback_sequence_at_start:
            return False, "waiting for fresh simulator feedback"
        measured = [self._actual_positions.get(name) for name in record.joint_names]
        if any(value is None for value in measured):
            return False, "waiting for requested-joint feedback"
        if record.resource == "gripper" and record.completion_policy == "grasp_range":
            opening = float(measured[0])
            if opening < 0.005:
                return False, "no_object_grasped" if self._now() - record.started_at >= GRIPPER_SETTLE_SEC else "waiting for gripper settle"
            in_tolerance = 0.005 <= opening <= 0.035
        else:
            tolerance = (
                GRIPPER_POSITION_TOLERANCE_M if record.resource == "gripper"
                else RAIL_POSITION_TOLERANCE_M if record.resource == "rail"
                else self.direct_target_position_tolerance
            )
            in_tolerance = all(
                abs(float(value) - record.targets[name]) <= tolerance
                for name, value in zip(record.joint_names, measured)
            )
        if not in_tolerance:
            return False, "target not reached"
        velocities = [self._actual_velocities.get(name) for name in record.joint_names]
        if any(value is not None and abs(float(value)) > VELOCITY_SETTLE_THRESHOLD for value in velocities):
            return False, "joint velocity not settled"
        if record.resource == "gripper" and self._now() - record.started_at < GRIPPER_SETTLE_SEC:
            return False, "waiting for gripper settle"
        return True, "target reached"

    def _check_direct_goals_locked(self) -> None:
        now = self._now()
        for record in list(self._direct_goals.values()):
            reached, reason = self._direct_goal_reached_locked(record)
            if reason == "no_object_grasped":
                self._finish_direct_locked(record, "no_object_grasped", reason)
            elif reached:
                record.stable_samples += 1
                if record.stable_samples >= STABLE_SAMPLES_REQUIRED:
                    self._finish_direct_locked(record, "reached", reason)
            else:
                record.stable_samples = 0
            timeout_sec = (
                self.rail_command_timeout_sec
                if record.resource == "rail"
                else DIRECT_TIMEOUT_SEC
            )
            if not record.done.is_set() and now - record.started_at >= timeout_sec:
                self._finish_direct_locked(
                    record, "timeout", f"{timeout_sec:g}s timeout: {reason}"
                )

    def _state1_callback(self, message: JointState) -> None:
        with self._lock:
            self._feedback_sequence += 1
            for index, name in enumerate(message.name):
                if index < len(message.position) and np.isfinite(float(message.position[index])): self._actual_positions[name] = float(message.position[index])
                if index < len(message.velocity) and np.isfinite(float(message.velocity[index])): self._actual_velocities[name] = float(message.velocity[index])
            self._check_direct_goals_locked()
        self._split_system1_state(message)

    def _state2_callback(self, message: JointState) -> None:
        self._split_system2_state(message)

    def _split_system1_state(self, message: JointState) -> None:
        rail, arm = JointState(header=message.header), JointState(header=message.header)
        for index, name in enumerate(message.name):
            target = rail if "rail" in name.lower() else arm if "panda" in name.lower() else None
            if target is None: continue
            target.name.append(name)
            if index < len(message.position): target.position.append(message.position[index])
            if index < len(message.velocity): target.velocity.append(message.velocity[index])
            if index < len(message.effort): target.effort.append(message.effort[index])
            if name == GRIPPER_JOINT and index < len(message.position): self.pub_gripper_state.publish(Float64(data=float(message.position[index])))
        if rail.name: self.pub_rail1_state.publish(rail)
        if arm.name: self.pub_franka1_state.publish(arm)

    def _split_system2_state(self, message: JointState) -> None:
        rail, arm = JointState(header=message.header), JointState(header=message.header)
        for index, name in enumerate(message.name):
            target = rail if "rail" in name.lower() else arm if "panda" in name.lower() else None
            if target is None: continue
            target.name.append(name)
            if index < len(message.position): target.position.append(message.position[index])
            if index < len(message.velocity): target.velocity.append(message.velocity[index])
            if index < len(message.effort): target.effort.append(message.effort[index])
        if rail.name: self.pub_rail2_state.publish(rail)
        if arm.name: self.pub_franka2_state.publish(arm)

    def _cartesian_joint_callback(self, message: JointState) -> None:
        with self._lock:
            if self._arm_owner() not in {"rrt", "direct_vertical"}:
                self.get_logger().warning("Blocked Cartesian joint stream: no Cartesian arm lease")
                return
            for index, name in enumerate(message.name):
                if self._is_arm_joint(name) and index < len(message.position):
                    self.sys1_cmd_cache[name] = float(message.position[index])
                    self.sys1_cmd_vel_cache[name] = float(message.velocity[index]) if index < len(message.velocity) else 0.0
                    self.sys1_cmd_received.add(name)

    def _safety_hold_callback(self, message: JointState) -> None:
        with self._lock:
            for index, name in enumerate(message.name):
                if self._is_arm_joint(name) and index < len(message.position):
                    self.sys1_cmd_cache[name], self.sys1_cmd_vel_cache[name] = float(message.position[index]), 0.0
                    self.sys1_cmd_received.add(name)
            self._arm_lease = None
            self._publish_direct_control_state()
            if (record := self._direct_goals.get("arm")) is not None: self._finish_direct_locked(record, "safety_interrupted", "RRT safety hold")

    def _legacy_joint_callback(self, _message: JointState) -> None:
        self.get_logger().warning("Rejected legacy /direct_joint_command; use /bridge/direct_joint_command action")

    def _legacy_gripper_callback(self, _message: Float64) -> None:
        self.get_logger().warning("Rejected legacy /gripper/command; use /bridge/direct_joint_command action")

    def _legacy_pose_callback(self, _message: Pose) -> None:
        self.get_logger().warning("Rejected legacy Cartesian pose topic; use RRT/vertical actions")

    def _sys2_command_callback(self, message: JointState) -> None:
        for index, name in enumerate(message.name):
            if index < len(message.position):
                self.sys2_cmd_cache[name] = float(message.position[index])
                self.sys2_cmd_received.add(name)
            if index < len(message.velocity): self.sys2_cmd_vel_cache[name] = float(message.velocity[index])

    def _damped_rail_command_locked(self) -> tuple[float, float] | None:
        """Return a smooth rail setpoint while preserving a 100 Hz keepalive."""
        if RAIL_JOINT not in self.sys1_cmd_received:
            return None
        target = float(self.sys1_cmd_cache[RAIL_JOINT])
        now = self._now()
        dt = min(max(now - self._last_rail_filter_update, 0.0), 0.25)
        self._last_rail_filter_update = now
        if self._filtered_rail_target is None:
            # Begin from feedback when possible, avoiding an initial command
            # jump if the bridge starts while the rail is already elsewhere.
            self._filtered_rail_target = float(self._actual_positions.get(RAIL_JOINT, target))
        requested_speed = abs(float(self.sys1_cmd_vel_cache.get(RAIL_JOINT, 0.0)))
        speed_limit = min(
            self.rail_damping_max_speed_mps,
            requested_speed if requested_speed > 1e-6 else self.rail_damping_max_speed_mps,
        )
        alpha = 1.0 - float(np.exp(-dt / self.rail_damping_time_constant_sec))
        desired_step = (target - self._filtered_rail_target) * alpha
        max_step = speed_limit * dt
        step = float(np.clip(desired_step, -max_step, max_step))
        self._filtered_rail_target += step
        velocity = 0.0 if dt <= 1e-6 else step / dt
        return self._filtered_rail_target, velocity

    def _publish_cached_command(self, publisher, positions, velocities, received, *, damp_rail: bool = False) -> None:
        if not received: return
        damped_rail = self._damped_rail_command_locked() if damp_rail else None
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        for name in received:
            message.name.append(name)
            if name == RAIL_JOINT and damped_rail is not None:
                message.position.append(float(damped_rail[0]))
                message.velocity.append(float(damped_rail[1]))
            else:
                message.position.append(float(positions[name]))
                message.velocity.append(float(velocities.get(name, 0.0)))
        publisher.publish(message)

    def _periodic_update(self) -> None:
        with self._lock:
            self._check_direct_goals_locked()
            self._publish_cached_command(
                self.pub_sys1_cmd, self.sys1_cmd_cache, self.sys1_cmd_vel_cache,
                self.sys1_cmd_received, damp_rail=True,
            )
            self._publish_cached_command(self.pub_sys2_cmd, self.sys2_cmd_cache, self.sys2_cmd_vel_cache, self.sys2_cmd_received)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = RailUnifiedBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
