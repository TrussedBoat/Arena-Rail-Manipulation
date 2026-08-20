#!/usr/bin/env python3
"""RRT motion-planning gateway for Panda end-effector pose commands.

The node owns the /rrt/execute_pose action, plans a self-collision-free joint
trajectory with MPlib, rejects joint-limit and singular configurations, and
publishes a tagged trajectory to the Panda controller. During execution it
monitors actual joint feedback and issues a direct hold command if a safety
condition is violated.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
import threading
import uuid

import mplib
import numpy as np
import rclpy
import ros_logger
ros_logger.setup_ros_logging()
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from interface.action import ExecuteRrtPose
from interface.msg import ControllerExecutionStatus, TaggedJointTrajectory
from interface.srv import AcquireArmLease, ReleaseArmLease


PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
PANDA_RTB_TO_URDF_EEF_OFFSET_M = 0.0216


class PlanningError(RuntimeError):
    """Raised when an EEF request cannot be converted into a safe trajectory."""


def _finite_pose(pose: Pose) -> bool:
    values = (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return all(math.isfinite(float(value)) for value in values)


def _normalised_wxyz(pose: Pose) -> np.ndarray:
    xyzw = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=float,
    )
    norm = float(np.linalg.norm(xyzw))
    if not math.isfinite(norm) or norm < 1e-9:
        raise PlanningError("Target orientation quaternion is invalid")
    xyzw /= norm
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def _duration(seconds: float) -> Duration:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    nanos = int(round((seconds - whole) * 1e9))
    if nanos >= 1_000_000_000:
        whole += 1
        nanos -= 1_000_000_000
    return Duration(sec=whole, nanosec=nanos)


class RRTPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("arena_rrt_planner")
        self._callbacks = ReentrantCallbackGroup()
        self._action_lock = threading.RLock()

        default_model_root = Path(
            os.environ.get(
                "ARENA_RRT_MODEL_ROOT",
                "/home/thinkstation-sim/workspace/home_robotics/"
                "homerobotics_ws/src/motion_planners/data/panda",
            )
        )
        self.declare_parameter("pose_cmd_topic", "/rrt/pose_command")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("trajectory_topic", "/rrt/tagged_joint_trajectory")
        self.declare_parameter("controller_execution_status_topic", "/panda/controller_execution_status")
        self.declare_parameter("controller_cancel_topic", "/panda/cancel_execution")
        self.declare_parameter("action_timeout_sec", 30.0)
        # This legacy pulse is arm-only; action completion is instead correlated
        # through /panda/controller_execution_status.
        self.declare_parameter("trajectory_complete_topic", "/panda/trajectory_complete")
        self.declare_parameter("status_topic", "/rrt/status")
        self.declare_parameter("ready_topic", "/rrt/ready")
        # Keep RRT safety holds separate from the orchestrator's direct-joint
        # command channel.  The bridge still relays this topic to Isaac Sim,
        # but its distinct name prevents a safety hold from looking like a
        # normal VLM joint command to diagnostics and ROS tooling.
        self.declare_parameter("hold_topic", "/rrt/hold_command")
        self.declare_parameter("cancel_topic", "/rrt/cancel")
        self.declare_parameter("reverse_topic", "/rrt/reverse_last_trajectory")
        self.declare_parameter("model_root", str(default_model_root))
        self.declare_parameter("move_group", "eef")
        self.declare_parameter("planning_time_sec", 5.0)
        self.declare_parameter("planning_attempts", 3)
        self.declare_parameter("time_step_sec", 0.01)
        self.declare_parameter("rrt_range_rad", 0.10)
        self.declare_parameter("edge_resolution_rad", 0.04)
        self.declare_parameter("joint_limit_margin_rad", 0.01)
        self.declare_parameter("singularity_warn", 0.10)
        self.declare_parameter("singularity_stop", 0.045)
        self.declare_parameter("goal_position_tolerance_m", 0.005)
        self.declare_parameter("goal_orientation_tolerance_rad", 0.01)
        self.declare_parameter("eef_min_z_m", 0.08)

        self.planning_time_sec = float(self.get_parameter("planning_time_sec").value)
        self.planning_attempts = int(self.get_parameter("planning_attempts").value)
        self.time_step_sec = float(self.get_parameter("time_step_sec").value)
        self.rrt_range_rad = float(self.get_parameter("rrt_range_rad").value)
        self.edge_resolution_rad = float(self.get_parameter("edge_resolution_rad").value)
        self.joint_limit_margin_rad = float(
            self.get_parameter("joint_limit_margin_rad").value
        )
        self.singularity_warn = float(self.get_parameter("singularity_warn").value)
        self.singularity_stop = float(self.get_parameter("singularity_stop").value)
        self.goal_position_tolerance_m = float(
            self.get_parameter("goal_position_tolerance_m").value
        )
        self.goal_orientation_tolerance_rad = float(
            self.get_parameter("goal_orientation_tolerance_rad").value
        )
        self.eef_min_z_m = float(self.get_parameter("eef_min_z_m").value)
        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)
        self._validate_parameters()

        model_root = Path(str(self.get_parameter("model_root").value)).expanduser()
        self._model_tempdir = tempfile.TemporaryDirectory(prefix="arena_rrt_model_")
        model_copy = Path(self._model_tempdir.name) / "panda"
        self._copy_model(model_root, model_copy)

        self.planner = mplib.Planner(
            urdf=model_copy / "panda.urdf",
            srdf=model_copy / "panda.srdf",
            move_group=str(self.get_parameter("move_group").value),
            new_package_keyword="",
            verbose=False,
        )
        self.robot = rtb.models.Panda()
        self.arm_limits = np.asarray(self.planner.joint_limits[:7], dtype=float)

        self.current_positions: dict[str, float] = {}
        self.active_trajectory: np.ndarray | None = None
        self.active_trajectory_message: JointTrajectory | None = None
        self.last_successful_trajectory: JointTrajectory | None = None
        self.executing = False
        self.safety_stop_sent = False
        self._last_runtime_warning = 0.0
        self._active_action_goal = None
        self._active_execution_id = None
        self._active_lease_id = None
        self._action_result_state = None
        self._action_result_reason = ""
        self._action_done = threading.Event()

        self.trajectory_pub = self.create_publisher(
            TaggedJointTrajectory, str(self.get_parameter("trajectory_topic").value), 10
        )
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.ready_pub = self.create_publisher(
            Bool, str(self.get_parameter("ready_topic").value), 10
        )
        self.hold_pub = self.create_publisher(
            JointState, str(self.get_parameter("hold_topic").value), 10
        )
        self.trajectory_complete_pub = self.create_publisher(
            Bool, str(self.get_parameter("trajectory_complete_topic").value), 10
        )
        self.controller_cancel_pub = self.create_publisher(
            String, str(self.get_parameter("controller_cancel_topic").value), 10
        )
        self.acquire_lease_client = self.create_client(
            AcquireArmLease, "/bridge/acquire_arm_lease", callback_group=self._callbacks
        )
        self.release_lease_client = self.create_client(
            ReleaseArmLease, "/bridge/release_arm_lease", callback_group=self._callbacks
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._joint_state_callback,
            10,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            Pose,
            str(self.get_parameter("pose_cmd_topic").value),
            self._pose_callback,
            10,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("trajectory_complete_topic").value),
            self._trajectory_complete_callback,
            10,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            Empty,
            str(self.get_parameter("cancel_topic").value),
            self._cancel_callback,
            10,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            Empty,
            str(self.get_parameter("reverse_topic").value),
            self._reverse_last_trajectory_callback,
            10,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            ControllerExecutionStatus,
            str(self.get_parameter("controller_execution_status_topic").value),
            self._controller_execution_status_callback,
            10,
            callback_group=self._callbacks,
        )
        self.execute_pose_action = ActionServer(
            self,
            ExecuteRrtPose,
            "/rrt/execute_pose",
            execute_callback=self._execute_pose_action,
            goal_callback=self._rrt_goal_callback,
            cancel_callback=self._rrt_cancel_callback,
            callback_group=self._callbacks,
        )
        self.ready_timer = self.create_timer(
            1.0, lambda: self.ready_pub.publish(Bool(data=True))
        )
        self.ready_pub.publish(Bool(data=True))

        self.get_logger().info(
            "RRT planner ready: /rrt/execute_pose -> lease -> validation -> tagged trajectory"
        )

    def _validate_parameters(self) -> None:
        positive = {
            "planning_time_sec": self.planning_time_sec,
            "time_step_sec": self.time_step_sec,
            "rrt_range_rad": self.rrt_range_rad,
            "edge_resolution_rad": self.edge_resolution_rad,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"RRT parameters must be positive: {', '.join(invalid)}")
        if self.joint_limit_margin_rad < 0:
            raise ValueError("joint_limit_margin_rad must be non-negative")
        if self.planning_attempts < 1:
            raise ValueError("planning_attempts must be at least one")
        if not 0 < self.singularity_stop < self.singularity_warn:
            raise ValueError("Require 0 < singularity_stop < singularity_warn")
        if not math.isfinite(self.eef_min_z_m) or self.eef_min_z_m < 0.0:
            raise ValueError("eef_min_z_m must be a finite non-negative height")

    @staticmethod
    def _copy_model(source: Path, destination: Path) -> None:
        required = (source / "panda.urdf", source / "panda.srdf")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing Panda planner model: " + ", ".join(missing))
        # MPlib writes a package-keyword-expanded URDF beside the input file,
        # so copy the read-only workspace model into a private writable directory.
        shutil.copytree(source, destination)

    def _publish_status(self, state: str, message: str, **details: object) -> None:
        payload = {
            "state": state,
            "message": message,
            "execution_id": self._active_execution_id or "",
            **details,
        }
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        log = self.get_logger().error if state == "failure" else self.get_logger().info
        log(f"[RRT {state.upper()}] {message}")

    def _wait_future(self, future, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.result() if future.done() else None

    def _acquire_rrt_lease(self, execution_id: str):
        if not self.acquire_lease_client.wait_for_service(timeout_sec=2.0):
            return None, "bridge lease service unavailable"
        request = AcquireArmLease.Request()
        request.owner, request.execution_id = "rrt", execution_id
        response = self._wait_future(self.acquire_lease_client.call_async(request), 2.0)
        if response is None:
            return None, "bridge lease request timed out"
        if not response.granted:
            return None, (
                f"{response.reason}; owner={response.active_owner} "
                f"execution={response.active_execution_id} age={response.owner_age_sec:.2f}s "
                f"remaining={response.owner_remaining_sec:.2f}s"
            )
        return str(response.lease_id), ""

    def _release_rrt_lease(self, lease_id: str | None, state: str, reason: str) -> None:
        if not lease_id or not self.release_lease_client.wait_for_service(timeout_sec=1.0):
            return
        request = ReleaseArmLease.Request()
        request.lease_id, request.terminal_state, request.reason = lease_id, state, reason
        self.release_lease_client.call_async(request)

    def _rrt_goal_callback(self, goal_request):
        return GoalResponse.ACCEPT if _finite_pose(goal_request.target_pose) else GoalResponse.REJECT

    def _rrt_cancel_callback(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _rrt_result(self, success: bool, state: str, reason: str, execution_id: str):
        result = ExecuteRrtPose.Result()
        result.success, result.state, result.reason, result.execution_id = success, state, reason, execution_id
        return result

    def _action_feedback(self, goal_handle, state: str, reason: str, started_at: float) -> None:
        feedback = ExecuteRrtPose.Feedback()
        feedback.state, feedback.reason = state, reason
        feedback.elapsed_sec = float(time.monotonic() - started_at)
        goal_handle.publish_feedback(feedback)

    def _controller_execution_status_callback(self, message: ControllerExecutionStatus) -> None:
        with self._action_lock:
            if not self.executing or str(message.execution_id) != self._active_execution_id:
                return
            state, reason = str(message.state), str(message.reason)
            if state == "reached":
                self.executing = False
                self.last_successful_trajectory = self.active_trajectory_message
                self.active_trajectory = None
                self.active_trajectory_message = None
                self._action_result_state, self._action_result_reason = "reached", reason
                self._publish_status("success", "RRT tagged trajectory completed")
                self._action_done.set()
            elif state in {"cancelled", "timeout", "failed", "safety_interrupted"}:
                self.executing = False
                self.active_trajectory = None
                self.active_trajectory_message = None
                self._action_result_state, self._action_result_reason = state, reason
                self._publish_status("failure", f"Controller execution {state}: {reason}")
                self._action_done.set()

    def _execute_pose_action(self, goal_handle):
        started_at = time.monotonic()
        execution_id = f"rrt-{uuid.uuid4().hex}"
        lease_id, reason = self._acquire_rrt_lease(execution_id)
        if lease_id is None:
            result = self._rrt_result(False, "blocked_by_direct_action", reason, execution_id)
            goal_handle.abort()
            return result
        try:
            with self._action_lock:
                if self.executing:
                    raise PlanningError("Planner is busy executing another trajectory")
                pose = goal_handle.request.target_pose
                if pose.position.z < self.eef_min_z_m:
                    raise PlanningError(
                        f"Target EEF z={pose.position.z:.4f}m is below the table safety floor of {self.eef_min_z_m:.4f}m"
                    )
                self._active_action_goal = goal_handle
                self._active_execution_id = execution_id
                self._active_lease_id = lease_id
                self._action_result_state, self._action_result_reason = None, ""
                self._action_done.clear()
                self._publish_status("planning", "Planning collision-free RRT trajectory")
                self._action_feedback(goal_handle, "planning", "lease granted", started_at)
                if self._target_already_reached(pose):
                    self._action_result_state, self._action_result_reason = "reached", "EEF target already reached"
                    self._action_done.set()
                else:
                    trajectory = self._plan(pose)
                    self.active_trajectory = np.asarray([point.positions for point in trajectory.points], dtype=float)
                    self.active_trajectory_message = trajectory
                    self.safety_stop_sent = False
                    self.executing = True
                    tagged = TaggedJointTrajectory()
                    tagged.execution_id, tagged.trajectory = execution_id, trajectory
                    self.trajectory_pub.publish(tagged)
                    self._publish_status("executing", "Validated tagged RRT trajectory published", points=len(trajectory.points))
            last_feedback = 0.0
            while not self._action_done.wait(timeout=0.05):
                now = time.monotonic()
                if goal_handle.is_cancel_requested:
                    self.controller_cancel_pub.publish(String(data=execution_id))
                    with self._action_lock:
                        self.executing = False
                        self.active_trajectory = None
                        self.active_trajectory_message = None
                        self._action_result_state, self._action_result_reason = "cancelled", "RRT action cancellation requested"
                        self._action_done.set()
                    break
                if now - started_at >= self.action_timeout_sec:
                    self.controller_cancel_pub.publish(String(data=execution_id))
                    with self._action_lock:
                        self.executing = False
                        self.active_trajectory = None
                        self.active_trajectory_message = None
                        self._action_result_state, self._action_result_reason = "timeout", "RRT planning/execution timeout"
                        self._action_done.set()
                    break
                if now - last_feedback >= 0.25:
                    self._action_feedback(goal_handle, "executing" if self.executing else "planning", "", started_at)
                    last_feedback = now
            state = self._action_result_state or "failed"
            reason = self._action_result_reason or "RRT action stopped unexpectedly"
        except Exception as exc:
            state, reason = "planning_failed", str(exc)
            self._publish_status("failure", reason)
        finally:
            self._release_rrt_lease(lease_id, state, reason)
            with self._action_lock:
                self._active_action_goal = None
                self._active_execution_id = None
                self._active_lease_id = None
        result = self._rrt_result(state == "reached", state, reason, execution_id)
        if state == "reached": goal_handle.succeed()
        elif state == "cancelled": goal_handle.canceled()
        else: goal_handle.abort()
        return result

    def _joint_state_callback(self, message: JointState) -> None:
        try:
            for name, position in zip(message.name, message.position):
                if math.isfinite(float(position)):
                    self.current_positions[name] = float(position)

            if not self.executing or self.safety_stop_sent:
                return
            q = self._current_arm_q(raise_if_missing=False)
            if q is None:
                return
            reason = self._configuration_error(q, runtime=True)
            if reason is not None:
                self._stop_with_hold(q, reason)
                return

            sigma = self._sigma_min(q)
            now = time.monotonic()
            if sigma < self.singularity_warn and now - self._last_runtime_warning >= 1.0:
                self._last_runtime_warning = now
                self.get_logger().warning(
                    f"RRT execution near singularity: sigma_min={sigma:.5f}"
                )
        except Exception as exc:
            self.get_logger().error(
                "Joint-state safety callback failed without terminating planner: "
                f"{exc}\n{traceback.format_exc()}"
            )
            if self.executing:
                q = self._current_arm_q(raise_if_missing=False)
                if q is not None:
                    self._stop_with_hold(q, f"safety monitor exception: {exc}")

    def _trajectory_complete_callback(self, message: Bool) -> None:
        if message.data:
            self.get_logger().debug(
                "Ignoring untagged /panda/trajectory_complete; tagged controller status owns action completion"
            )

    def _cancel_callback(self, _message: Empty) -> None:
        self.get_logger().warning("Rejected legacy /rrt/cancel; cancel /rrt/execute_pose action instead")
        return
        # Legacy implementation retained below for source-history readability.
        if not self.executing:
            self._publish_status("cancelled", "No active RRT trajectory to cancel")
            return
        q = self._current_arm_q(raise_if_missing=False)
        if q is None:
            self._publish_status(
                "failure", "Cannot cancel RRT trajectory without current joint telemetry"
            )
            return
        self._publish_hold(q)
        self.executing = False
        self.active_trajectory = None
        self.active_trajectory_message = None
        self.safety_stop_sent = False
        self._publish_status("cancelled", "RRT trajectory cancelled with joint hold")

    def _reverse_last_trajectory_callback(self, _message: Empty) -> None:
        """Keep the old debug trigger visible without bypassing ownership.

        A reverse move needs its own recoverable action so its lease, completion,
        and cancellation are correlated like every other arm motion.  Until that
        action exists, rejecting this legacy trigger is safer than publishing an
        unowned trajectory.
        """
        self._publish_status(
            "failure",
            "Legacy reverse is disabled by central command ownership; use a recovery action",
        )

    @staticmethod
    def _reverse_trajectory(trajectory: JointTrajectory) -> JointTrajectory:
        total_sec = (
            float(trajectory.points[-1].time_from_start.sec)
            + float(trajectory.points[-1].time_from_start.nanosec) * 1e-9
        )
        reversed_trajectory = JointTrajectory()
        reversed_trajectory.joint_names = list(trajectory.joint_names)
        for point in reversed(trajectory.points):
            point_sec = float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9
            reversed_point = JointTrajectoryPoint()
            reversed_point.positions = list(point.positions)
            if point.velocities:
                reversed_point.velocities = [-float(value) for value in point.velocities]
            if point.accelerations:
                reversed_point.accelerations = list(point.accelerations)
            reversed_point.time_from_start = _duration(total_sec - point_sec)
            reversed_trajectory.points.append(reversed_point)
        return reversed_trajectory

    def _current_arm_q(self, *, raise_if_missing: bool = True) -> np.ndarray | None:
        missing = [name for name in PANDA_JOINT_NAMES if name not in self.current_positions]
        if missing:
            if raise_if_missing:
                raise PlanningError("Missing joint telemetry: " + ", ".join(missing))
            return None
        return np.array([self.current_positions[name] for name in PANDA_JOINT_NAMES])

    def _current_full_q(self) -> np.ndarray:
        arm = self._current_arm_q()
        finger1 = self.current_positions.get("panda_finger_joint1", 0.02)
        finger2 = self.current_positions.get("panda_finger_joint2", finger1)
        return np.concatenate([arm, [finger1, finger2]])

    def _pose_callback(self, pose: Pose) -> None:
        self.get_logger().warning("Rejected legacy /rrt/pose_command; use /rrt/execute_pose action")
        return
        # Legacy implementation retained below for source-history readability.
        try:
            if self.executing:
                self._publish_status("failure", "Planner is busy executing another trajectory")
                return
            if not _finite_pose(pose):
                raise PlanningError("Target pose contains a non-finite value")
            if pose.position.z < self.eef_min_z_m:
                raise PlanningError(
                    f"Target EEF z={pose.position.z:.4f}m is below the table safety "
                    f"floor of {self.eef_min_z_m:.4f}m"
                )
            self._publish_status("planning", "Planning collision-free RRT trajectory")
            if self._target_already_reached(pose):
                self._publish_status("success", "EEF target is already reached")
                self.trajectory_complete_pub.publish(Bool(data=True))
                return
            trajectory = self._plan(pose)
        except Exception as exc:
            self.get_logger().error(
                "Pose request failed without terminating planner: "
                f"{exc}\n{traceback.format_exc()}"
            )
            self._publish_status("failure", str(exc))
            return

        self.active_trajectory = np.asarray(
            [point.positions for point in trajectory.points], dtype=float
        )
        self.active_trajectory_message = trajectory
        self.safety_stop_sent = False
        self.executing = True
        tagged = TaggedJointTrajectory()
        tagged.execution_id = self._active_execution_id or f"legacy-{uuid.uuid4().hex}"
        tagged.trajectory = trajectory
        self.trajectory_pub.publish(tagged)
        self._publish_status(
            "executing",
            "Validated RRT trajectory published",
            points=len(trajectory.points),
        )

    def _target_already_reached(self, pose: Pose) -> bool:
        if not _finite_pose(pose):
            return False
        q = self._current_arm_q(raise_if_missing=False)
        if q is None:
            return False
        current = self._eef_transform(q)

        target_position = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=float
        )
        position_error = float(np.linalg.norm(target_position - current[:3, 3]))
        target_rotation = Rotation.from_quat(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )
        current_rotation = Rotation.from_matrix(current[:3, :3])
        orientation_error = float(
            np.linalg.norm((target_rotation * current_rotation.inv()).as_rotvec())
        )
        return (
            position_error <= self.goal_position_tolerance_m
            and orientation_error <= self.goal_orientation_tolerance_rad
        )

    def _plan(self, pose: Pose) -> JointTrajectory:
        if not _finite_pose(pose):
            raise PlanningError("Target pose contains a non-finite value")
        if pose.position.z < self.eef_min_z_m:
            raise PlanningError(
                f"Target EEF z={pose.position.z:.4f}m is below the table safety "
                f"floor of {self.eef_min_z_m:.4f}m"
            )
        quaternion = _normalised_wxyz(pose)
        start_q = self._current_full_q()
        start_error = self._configuration_error(start_q[:7])
        if start_error is not None:
            raise PlanningError(f"Current configuration is unsafe: {start_error}")

        target = mplib.Pose(
            p=[pose.position.x, pose.position.y, pose.position.z],
            q=quaternion,
        )
        failures: list[str] = []
        for attempt in range(1, self.planning_attempts + 1):
            result = self.planner.plan_pose(
                target,
                start_q,
                time_step=self.time_step_sec,
                rrt_range=self.rrt_range_rad,
                planning_time=self.planning_time_sec,
                fix_joint_limits=True,
                wrt_world=False,
                simplify=True,
                verbose=False,
            )
            if result.get("status") != "Success":
                reason = f"planner status: {result.get('status', 'unknown error')}"
                failures.append(reason)
                self.get_logger().warning(
                    f"RRT attempt {attempt}/{self.planning_attempts} failed: {reason}"
                )
                continue

            try:
                positions = np.asarray(result.get("position"), dtype=float)
                if positions.ndim != 2 or positions.shape[1] < 7 or len(positions) == 0:
                    raise PlanningError("planner returned an invalid joint trajectory")
                arm_positions = positions[:, :7]
                minimum_sigma = self._validate_path(arm_positions)
            except PlanningError as exc:
                failures.append(str(exc))
                self.get_logger().warning(
                    f"RRT attempt {attempt}/{self.planning_attempts} rejected: {exc}"
                )
                continue

            self.get_logger().info(
                f"RRT validation passed on attempt {attempt}: "
                f"{len(arm_positions)} points, minimum sigma={minimum_sigma:.5f}"
            )
            return self._trajectory_message(result, arm_positions)

        last_reason = failures[-1] if failures else "unknown planner failure"
        raise PlanningError(
            f"No safe RRT path after {self.planning_attempts} attempts; "
            f"last rejection: {last_reason}"
        )

    def _configuration_error(self, q: np.ndarray, *, runtime: bool = False) -> str | None:
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            return "invalid seven-joint configuration"
        lower = self.arm_limits[:, 0] + self.joint_limit_margin_rad
        upper = self.arm_limits[:, 1] - self.joint_limit_margin_rad
        if np.any(q < lower) or np.any(q > upper):
            index = int(np.flatnonzero((q < lower) | (q > upper))[0])
            return f"{PANDA_JOINT_NAMES[index]} violates its safety-margin limit"
        collisions = self.planner.check_for_self_collision(q)
        if collisions:
            return f"self-collision detected ({collisions[0]})"
        eef_z = float(self._eef_transform(q)[2, 3])
        if eef_z < self.eef_min_z_m:
            scope = "runtime" if runtime else "planned"
            return (
                f"{scope} EEF z={eef_z:.4f}m is below the table safety floor "
                f"of {self.eef_min_z_m:.4f}m"
            )
        sigma = self._sigma_min(q)
        if sigma <= self.singularity_stop:
            scope = "runtime" if runtime else "planned"
            return (
                f"{scope} singularity threshold violated: sigma_min={sigma:.5f} "
                f"<= {self.singularity_stop:.5f}"
            )
        return None

    def _eef_transform(self, q: np.ndarray) -> np.ndarray:
        """Return the planner EEF transform in panda_link0 coordinates."""
        transform = np.asarray(self.robot.fkine(q).A, dtype=float)
        # Robotics Toolbox ends at its 0.1034m tool frame while the planning
        # URDF defines EEF at 0.125m, leaving a 0.0216m local-Z offset.
        transform[:3, 3] += transform[:3, 2] * PANDA_RTB_TO_URDF_EEF_OFFSET_M
        return transform

    def _sigma_min(self, q: np.ndarray) -> float:
        jacobian = np.asarray(self.robot.jacob0(q), dtype=float)
        return float(np.linalg.svd(jacobian, compute_uv=False)[-1])

    def _validate_path(self, path: np.ndarray) -> float:
        minimum_sigma = math.inf
        for index in range(len(path)):
            previous = path[index - 1] if index > 0 else path[index]
            delta = path[index] - previous
            samples = max(1, int(math.ceil(np.max(np.abs(delta)) / self.edge_resolution_rad)))
            for fraction in np.linspace(0.0, 1.0, samples + 1)[1:]:
                q = previous + delta * fraction
                error = self._configuration_error(q)
                if error is not None:
                    raise PlanningError(f"Unsafe path near waypoint {index}: {error}")
                minimum_sigma = min(minimum_sigma, self._sigma_min(q))
        return float(minimum_sigma)

    def _trajectory_message(
        self, result: dict[str, object], positions: np.ndarray
    ) -> JointTrajectory:
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.header.frame_id = "panda_link0"
        trajectory.joint_names = PANDA_JOINT_NAMES

        velocities = np.asarray(result.get("velocity", []), dtype=float)
        accelerations = np.asarray(result.get("acceleration", []), dtype=float)
        times = np.asarray(result.get("time", []), dtype=float)
        for index, q in enumerate(positions):
            point = JointTrajectoryPoint()
            point.positions = q.tolist()
            if velocities.shape[0] == len(positions) and velocities.shape[1] >= 7:
                point.velocities = velocities[index, :7].tolist()
            else:
                point.velocities = [0.0] * 7
            if accelerations.shape[0] == len(positions) and accelerations.shape[1] >= 7:
                point.accelerations = accelerations[index, :7].tolist()
            if times.shape == (len(positions),):
                point.time_from_start = _duration(float(times[index]))
            else:
                point.time_from_start = _duration(index * self.time_step_sec)
            trajectory.points.append(point)
        return trajectory

    def _stop_with_hold(self, q: np.ndarray, reason: str) -> None:
        self._publish_hold(q)
        if self._active_execution_id:
            self.controller_cancel_pub.publish(String(data=self._active_execution_id))
        self.safety_stop_sent = True
        self.executing = False
        self.active_trajectory = None
        self.active_trajectory_message = None
        self._action_result_state = "safety_interrupted"
        self._action_result_reason = reason
        self._action_done.set()
        self._publish_status("failure", f"Runtime safety stop: {reason}")

    def _publish_hold(self, q: np.ndarray) -> None:
        hold = JointState()
        hold.header.stamp = self.get_clock().now().to_msg()
        hold.name = PANDA_JOINT_NAMES
        hold.position = np.asarray(q, dtype=float).tolist()
        hold.velocity = [0.0] * 7
        self.hold_pub.publish(hold)

    def destroy_node(self):
        try:
            return super().destroy_node()
        finally:
            self._model_tempdir.cleanup()


def main() -> int:
    rclpy.init()
    node = None
    try:
        node = RRTPlannerNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
