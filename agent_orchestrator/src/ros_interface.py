import base64
import copy
import json
import math
import threading
import time
import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
from std_msgs.msg import Bool, Empty, Float64, String
import tf2_ros
from geometry_msgs.msg import Pose
from interface.action import FindObject
from interface.msg import DetectedObjectArray
from scipy.spatial.transform import Rotation

_shared_node = None
VLM_CAMERA_MAX_SIDE_PX = 256
GRIPPER_COMMAND_CLOSED = 0.0
GRIPPER_COMMAND_OPEN = 0.04
GRIPPER_COMMAND_MAX = 0.08

def get_shared_node():
    global _shared_node
    if _shared_node is None:
        _shared_node = RobotHardwareInterface()
        import threading
        threading.Thread(target=rclpy.spin, args=(_shared_node,), daemon=True).start()
    return _shared_node

# ── 1. ROS 2 INTEGRATION LAYER ──────────────────────────────────────────────
class RobotHardwareInterface(Node):
    def __init__(self):
        super().__init__('agent_hardware_interface')
        self.image_sub = self.create_subscription(Image, '/sim/rail_franka1/cam/wrist/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/sim/rail_franka1/cam/wrist/depth/image_raw', self.depth_callback, 10)
        self.rail_subscriber = self.create_subscription(JointState, '/sim/rail_franka1/joint_states', self.rail_state_callback, 10)
        self.rail_publisher = self.create_publisher(JointState, '/direct_joint_command', 10)
        self.gripper_publisher = self.create_publisher(Float64, '/gripper/command', 10)
        self.gripper_state_sub = self.create_subscription(
            Float64, '/gripper_state', self.gripper_state_callback, 10
        )
        self.pose_publisher = self.create_publisher(Pose, '/rrt/pose_command', 10)
        self.rrt_cancel_publisher = self.create_publisher(Empty, '/rrt/cancel', 10)
        self.rrt_reverse_publisher = self.create_publisher(Empty, '/rrt/reverse_last_trajectory', 10)
        self.controller_state_sub = self.create_subscription(
            Bool, '/panda/controller_state', self.controller_state_callback, 10
        )
        self.controller_ready_sub = self.create_subscription(
            Bool, '/panda/controller_ready', self.controller_ready_callback, 10
        )
        direct_control_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.direct_joint_control_sub = self.create_subscription(
            Bool,
            '/panda/direct_joint_control_active',
            self.direct_joint_control_callback,
            direct_control_qos,
        )
        self.rrt_status_sub = self.create_subscription(
            String, '/rrt/status', self.rrt_status_callback, 10
        )
        self.rrt_ready_sub = self.create_subscription(
            Bool, '/rrt/ready', self.rrt_ready_callback, 10
        )
        semantic_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.semantic_objects_sub = self.create_subscription(
            DetectedObjectArray, '/semantic/objects', self.semantic_objects_callback, semantic_qos
        )
        self.semantic_find_client = ActionClient(self, FindObject, '/semantic/find_object')

        self.grasp_srv = self.create_service(Trigger, '/vlm_grasp_completed', self.grasp_callback)
        self.place_srv = self.create_service(Trigger, '/vlm_place_completed', self.place_callback)
        self.is_grasped = False
        self.is_placed = False
        
        self.bridge = CvBridge()
        self.latest_b64_image = None
        self.latest_vlm_b64_image = None
        self.latest_vlm_image_sequence = 0
        self.latest_depth_image = None  # raw numpy float32 depth frame (metres)
        self.current_rail_position = None
        self.current_panda_joint1 = None
        self.current_panda_joint6 = None
        self.current_joint_positions = {}
        self.current_gripper_state = None
        self.cartesian_controller_ready = False
        self.cartesian_control_active = False
        self.direct_joint_control_active = False
        self._cartesian_completion = threading.Event()
        self._cartesian_failure = None
        self._rrt_status_event = threading.Event()
        self._rrt_status_state = None
        self._rrt_status_message = None
        self._rrt_ready_at = None
        self._semantic_lock = threading.RLock()
        self._semantic_objects = {}
        self._semantic_objects_received = False
        self._semantic_candidate = None
        self._semantic_goal_handle = None
        self._semantic_result = None
        self._semantic_error = None

        # TF listener for camera→base transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def controller_ready_callback(self, msg):
        self.cartesian_controller_ready = bool(msg.data)

    def controller_state_callback(self, msg):
        # This is a one-shot completion notification from the Panda controller,
        # not an ownership state.  Ownership is tracked from /rrt/status so a
        # delayed completion pulse cannot re-lock the gripper after success.
        del msg

    def gripper_state_callback(self, msg):
        value = float(msg.data)
        if math.isfinite(value):
            self.current_gripper_state = value

    def direct_joint_control_callback(self, msg):
        self.direct_joint_control_active = bool(msg.data)

    def rrt_status_callback(self, msg):
        try:
            status = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        state = str(status.get('state', ''))
        self._rrt_status_state = state
        self._rrt_status_message = str(status.get('message', ''))
        self._rrt_status_event.set()
        if state in {'success', 'cancelled'}:
            # /panda/controller_state is a completion pulse, not a persistent
            # ownership signal.  RRT's terminal status is the authoritative
            # indication that its Cartesian command stream is no longer active.
            self.cartesian_control_active = False
        if state == 'failure':
            self._cartesian_failure = str(status.get('message', 'RRT planning failed'))
            self._cartesian_completion.set()
        elif state == 'success':
            self._cartesian_completion.set()
        elif state == 'cancelled':
            self._cartesian_completion.set()

    def rrt_ready_callback(self, msg):
        if msg.data:
            self._rrt_ready_at = time.monotonic()

    def semantic_objects_callback(self, message):
        with self._semantic_lock:
            self._semantic_objects_received = True
            self._semantic_objects = {
                item.object_id: copy.deepcopy(item) for item in message.objects
            }

    def start_semantic_find_object(self, target_class, min_confidence, timeout_sec):
        """Start one semantic FindObject action; callbacks run on the spin thread."""
        if not self.semantic_find_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Semantic perception action /semantic/find_object is unavailable")
        goal = FindObject.Goal()
        goal.target_class = str(target_class)
        goal.min_confidence = float(min_confidence)
        goal.timeout_sec = float(timeout_sec)
        with self._semantic_lock:
            self._semantic_candidate = None
            self._semantic_goal_handle = None
            self._semantic_result = None
            self._semantic_error = None
        future = self.semantic_find_client.send_goal_async(goal, feedback_callback=self._semantic_feedback)
        future.add_done_callback(self._semantic_goal_response)

    def _semantic_feedback(self, feedback_message):
        candidate = feedback_message.feedback.candidate
        with self._semantic_lock:
            self._semantic_candidate = copy.deepcopy(candidate) if candidate.object_id else None

    def _semantic_goal_response(self, future):
        try:
            handle = future.result()
            if not handle.accepted:
                raise RuntimeError("Semantic FindObject goal was rejected")
            with self._semantic_lock:
                self._semantic_goal_handle = handle
            result_future = handle.get_result_async()
            result_future.add_done_callback(self._semantic_result_callback)
        except Exception as exc:
            with self._semantic_lock:
                self._semantic_error = str(exc)

    def _semantic_result_callback(self, future):
        try:
            response = future.result().result
            with self._semantic_lock:
                self._semantic_result = copy.deepcopy(response)
        except Exception as exc:
            with self._semantic_lock:
                self._semantic_error = str(exc)

    def semantic_find_status(self):
        with self._semantic_lock:
            return {
                "candidate": copy.deepcopy(self._semantic_candidate),
                "result": copy.deepcopy(self._semantic_result),
                "error": self._semantic_error,
                "objects_received": self._semantic_objects_received,
            }

    def semantic_object(self, object_id):
        with self._semantic_lock:
            item = self._semantic_objects.get(str(object_id))
            return copy.deepcopy(item) if item is not None else None

    def cancel_semantic_find_object(self):
        with self._semantic_lock:
            handle = self._semantic_goal_handle
        if handle is not None:
            handle.cancel_goal_async()

    def wait_for_cartesian_controller(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if (
                self._rrt_ready_at is not None
                and time.monotonic() - self._rrt_ready_at <= 2.5
                and self.pose_publisher.get_subscription_count() > 0
                and not self.direct_joint_control_active
            ):
                return True
            time.sleep(0.05)
        return False

    def send_eef_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        timeout_sec: float = 30.0,
        readiness_timeout_sec: float = 5.0,
        tf_timeout_sec: float = 3.0,
        base_frame: str = 'panda_link0',
        eef_frame: str = 'eef',
    ) -> bool:
        self.begin_eef_pose(
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            readiness_timeout_sec=readiness_timeout_sec,
            tf_timeout_sec=tf_timeout_sec,
            base_frame=base_frame,
            eef_frame=eef_frame,
        )
        completed = self._cartesian_completion.wait(timeout=max(0.0, timeout_sec))
        if self._cartesian_failure is not None:
            raise RuntimeError(self._cartesian_failure)
        return completed and self._rrt_status_state == 'success'

    def begin_eef_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        readiness_timeout_sec: float = 5.0,
        tf_timeout_sec: float = 3.0,
        base_frame: str = 'panda_link0',
        eef_frame: str = 'eef',
    ) -> None:
        """Publish an RRT EEF target and return immediately."""
        values = (x, y, z, roll, pitch, yaw)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("EEF pose values must all be finite numbers")
        if not self.wait_for_cartesian_controller(readiness_timeout_sec):
            raise RuntimeError("RRT planner is not healthy or subscribed to /rrt/pose_command")

        # The home_robotics controller interprets Pose relative to panda_link0.
        quaternion = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        norm = float(np.linalg.norm(quaternion))
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("RPY values produced an invalid orientation quaternion")
        quaternion /= norm

        # Validate that the controller's feedback transform exists before actuating.
        self.get_transform_matrix(base_frame, eef_frame, timeout_sec=tf_timeout_sec)

        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = map(float, (x, y, z))
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = map(
            float, quaternion
        )

        self._cartesian_completion.clear()
        self._cartesian_failure = None
        self._rrt_status_event.clear()
        self._rrt_status_state = None
        self._rrt_status_message = None
        self.cartesian_control_active = True
        self.pose_publisher.publish(msg)
        self.get_logger().info(
            f"Published EEF pose in {base_frame}: "
            f"xyz=({x:.4f}, {y:.4f}, {z:.4f}), "
            f"rpy=({roll:.4f}, {pitch:.4f}, {yaw:.4f})"
        )

    def eef_motion_done(self) -> bool:
        return self._cartesian_completion.is_set()

    def wait_for_eef_motion(self, timeout_sec: float) -> bool:
        completed = self._cartesian_completion.wait(timeout=max(0.0, timeout_sec))
        if self._cartesian_failure is not None:
            raise RuntimeError(self._cartesian_failure)
        return completed and self._rrt_status_state == 'success'

    def cancel_eef_motion(self, timeout_sec: float = 3.0) -> bool:
        self._rrt_status_event.clear()
        self._rrt_status_state = None
        self._rrt_status_message = None
        self.rrt_cancel_publisher.publish(Empty())
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if self._rrt_status_event.wait(timeout=max(0.0, remaining)):
                if self._rrt_status_state == 'cancelled':
                    return True
                if self._rrt_status_state == 'failure':
                    raise RuntimeError(self._rrt_status_message or 'RRT cancellation failed')
                self._rrt_status_event.clear()
        return False

    def reverse_last_eef_motion(self, timeout_sec: float) -> bool:
        """Replay the most recently completed RRT trajectory in reverse."""
        if not self.wait_for_cartesian_controller(timeout_sec=min(5.0, timeout_sec)):
            raise RuntimeError("RRT planner is not ready for reverse trajectory replay")
        self._cartesian_completion.clear()
        self._cartesian_failure = None
        self._rrt_status_event.clear()
        self._rrt_status_state = None
        self._rrt_status_message = None
        self.cartesian_control_active = True
        self.rrt_reverse_publisher.publish(Empty())
        return self.wait_for_eef_motion(timeout_sec)

    def rrt_motion_active(self) -> bool:
        """Whether the latest planner status still owns a Cartesian trajectory."""
        return self._rrt_status_state in {"planning", "executing"}

    def ensure_rrt_idle(self, timeout_sec: float = 3.0) -> bool:
        """Cancel a previous Cartesian trajectory before direct motion/new RRT work."""
        if not self.rrt_motion_active():
            return True
        return self.cancel_eef_motion(timeout_sec=timeout_sec)

    def _ensure_direct_arm_control_allowed(self):
        if self.cartesian_control_active:
            self.get_logger().warning(
                "Direct Panda command requested; bridge arbitration will give it priority "
                "over Cartesian commands."
            )

    def grasp_callback(self, request, response):
        self.get_logger().info("[TELEMETRY] Robot announced grasp completion!")
        self.is_grasped = True
        
        response.success = True
        response.message = "VLM received grasp confirmation."
        return response

    def place_callback(self, request, response):
        self.get_logger().info("[TELEMETRY] Robot announced place completion!")
        self.is_placed = True
        
        response.success = True
        response.message = "VLM received place confirmation."
        return response

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            encoded, buffer = cv2.imencode(
                ".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not encoded:
                raise RuntimeError("OpenCV could not JPEG-encode the camera frame")
            self.latest_b64_image = base64.b64encode(buffer).decode('utf-8')

            height, width = cv_img.shape[:2]
            longest_side = max(height, width)
            vlm_img = cv_img
            if longest_side > VLM_CAMERA_MAX_SIDE_PX:
                scale = VLM_CAMERA_MAX_SIDE_PX / longest_side
                vlm_img = cv2.resize(
                    cv_img,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            encoded, buffer = cv2.imencode(
                ".jpg", vlm_img, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not encoded:
                raise RuntimeError("OpenCV could not JPEG-encode the VLM camera frame")
            self.latest_vlm_b64_image = base64.b64encode(buffer).decode('utf-8')
            self.latest_vlm_image_sequence += 1
        except Exception as e:
            self.get_logger().error(f"Image processing exception: {e}")

    def depth_callback(self, msg):
        """Store latest depth frame as a float32 numpy array (metres)."""
        try:
            # 32FC1 encoding → values already in metres
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth image processing exception: {e}")

    def get_latest_depth_image(self, timeout_sec: float = 3.0):
        """Block until a fresh depth frame is available and return it."""
        self.latest_depth_image = None
        start = time.time()
        while time.time() - start < timeout_sec:
            if self.latest_depth_image is not None:
                return self.latest_depth_image
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for depth image")

    def get_transform_matrix(self, target_frame: str, source_frame: str, timeout_sec: float = 3.0) -> np.ndarray:
        """Return a 4x4 homogeneous transform from source_frame to target_frame."""
        import rclpy.time
        from scipy.spatial.transform import Rotation
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                t = tf_stamped.transform.translation
                q = tf_stamped.transform.rotation
                rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                mat = np.eye(4)
                mat[:3, :3] = rot
                mat[:3, 3] = [t.x, t.y, t.z]
                return mat
            except Exception:
                time.sleep(0.1)
        raise TimeoutError(
            f"TF lookup timed out: {source_frame} → {target_frame} after {timeout_sec}s"
        )

    def rail_state_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            self.current_joint_positions[name] = float(position)

        if 'rail_j1' in msg.name:
            try:
                idx = msg.name.index('rail_j1')
                self.current_rail_position = msg.position[idx]
                
            except ValueError:
                pass
                
        if 'panda_joint1' in msg.name:
            try:
                idx = msg.name.index('panda_joint1')
                self.current_panda_joint1 = msg.position[idx]
            except ValueError:
                pass
                
        if 'panda_joint6' in msg.name:
            try:
                idx = msg.name.index('panda_joint6')
                self.current_panda_joint6 = msg.position[idx]
            except ValueError:
                pass

    def send_absolute_rail_command(self, absolute_value: float, speed: float = None):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['rail_j1']
        msg.position = [float(absolute_value)]
        if speed is not None:
            msg.velocity = [float(speed)]
        self.rail_publisher.publish(msg)
        self.get_logger().info(f"Published Absolute JointState for rail_j1: {absolute_value}m")
        
    def send_rail_and_joint6_command(self, rail_val: float, rail_speed: float, j6_val: float, j6_speed: float):
        self._ensure_direct_arm_control_allowed()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['rail_j1', 'panda_joint6']
        msg.position = [float(rail_val), float(j6_val)]
        msg.velocity = [float(rail_speed), float(j6_speed)]
        self.rail_publisher.publish(msg)
        self.get_logger().info(f"Published JointState for rail_j1={rail_val}m and panda_joint6={j6_val}rad")

    def send_panda_joint1_command(self, target_rad: float):
        self._ensure_direct_arm_control_allowed()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['panda_joint1']
        msg.position = [float(target_rad)]
        self.rail_publisher.publish(msg)
        self.get_logger().info(f"Published JointState for panda_joint1: {target_rad}rad")

    def send_panda_search_posture(self, j1: float, j2: float, j3: float, j4: float, j5: float, j6: float, j7: float):
        self._ensure_direct_arm_control_allowed()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        msg.position = [float(j1), float(j2), float(j3), float(j4), float(j5), float(j6), float(j7)]
        self.rail_publisher.publish(msg)
        self.get_logger().info("Published JointState for full arm search posture.")

    def set_gripper_command(self, opening_command: float) -> None:
        """Command the gripper opening through the independent direct interface.

        The Arena bridge forwards this value to ``panda_finger_joint1`` in
        metres: 0 is closed and 0.04 is normally fully open.  The bridge rejects direct gripper commands
        while an RRT Cartesian trajectory owns the Panda joints.
        """
        command = float(opening_command)
        if not math.isfinite(command):
            raise ValueError("Gripper command must be finite")
        if not GRIPPER_COMMAND_CLOSED <= command <= GRIPPER_COMMAND_MAX:
            raise ValueError(
                "Gripper command must be within "
                f"[{GRIPPER_COMMAND_CLOSED:g}, {GRIPPER_COMMAND_MAX:g}] metres, got {command:g}"
            )
        if self.cartesian_control_active:
            raise RuntimeError(
                "Cannot command gripper while an RRT Cartesian trajectory owns the Panda"
            )
        msg = Float64()
        msg.data = command
        self.gripper_publisher.publish(msg)
        self.get_logger().info(f"Published gripper command: opening={msg.data:.1f}")

    def open_gripper(self, opening_command: float = GRIPPER_COMMAND_OPEN) -> None:
        self.set_gripper_command(opening_command)

    def close_gripper(self, closing_command: float = GRIPPER_COMMAND_CLOSED) -> None:
        self.set_gripper_command(closing_command)

    def wait_for_gripper_target(
        self, target_state: float, tolerance: float = 0.005, timeout_sec: float = 5.0
    ) -> bool:
        target = float(target_state)
        if not math.isfinite(target) or tolerance <= 0.0 or timeout_sec <= 0.0:
            raise ValueError("Gripper target, tolerance, and timeout must be valid")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            current = self.current_gripper_state
            if current is not None and abs(float(current) - target) <= tolerance:
                return True
            time.sleep(0.02)
        return False

    def send_panda_joint6_command(self, target_rad: float, speed: float = None):
        self._ensure_direct_arm_control_allowed()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['panda_joint6']
        msg.position = [float(target_rad)]
        if speed is not None:
            msg.velocity = [float(speed)]
        self.rail_publisher.publish(msg)
        self.get_logger().info(f"Published JointState for panda_joint6: {target_rad}rad")

def wait_for_joint_target(node: RobotHardwareInterface, joint_name: str, target_value: float, tolerance=0.02, timeout=20.0) -> bool:
    start = time.time()
    while (time.time() - start) < timeout:
        # The background ROS thread updates the complete joint telemetry map.
        # Keep the legacy named fields as a fallback for compatibility.
        current_val = node.current_joint_positions.get(joint_name)
        if current_val is None:
            if joint_name == 'rail_j1':
                current_val = node.current_rail_position
            elif joint_name == 'panda_joint1':
                current_val = node.current_panda_joint1
            elif joint_name == 'panda_joint6':
                current_val = node.current_panda_joint6

        if current_val is not None and abs(current_val - target_value) <= tolerance:
            return True
            
        time.sleep(0.05)  # Yield to prevent maxing out the CPU loop
        
    return False

def wait_for_grasp(node: RobotHardwareInterface, timeout=60.0) -> bool:
    """Waits for the background thread to receive the 'grasped' message."""
    start = time.time()
    while (time.time() - start) < timeout:
        if node.is_grasped:
            return True
        time.sleep(0.1)
    return False

def wait_for_place(node: RobotHardwareInterface, timeout=60.0) -> bool:
    """Waits for the background thread to receive the 'placed' message."""
    start = time.time()
    while (time.time() - start) < timeout:
        if node.is_placed:
            return True
        time.sleep(0.1)
    return False
