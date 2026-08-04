import base64
import time
import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
from std_msgs.msg import String
import tf2_ros
from geometry_msgs.msg import PointStamped

_shared_node = None

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
        self.rail_publisher = self.create_publisher(JointState, '/joint_position_command', 10)

        self.grasp_srv = self.create_service(Trigger, '/vlm_grasp_completed', self.grasp_callback)
        self.place_srv = self.create_service(Trigger, '/vlm_place_completed', self.place_callback)
        self.is_grasped = False
        self.is_placed = False
        
        self.bridge = CvBridge()
        self.latest_b64_image = None
        self.latest_depth_image = None  # raw numpy float32 depth frame (metres)
        self.current_rail_position = None
        self.current_panda_joint1 = None
        self.current_panda_joint6 = None
        self.initial_rail_position = None

        # TF listener for camera→base transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
            _, buffer = cv2.imencode('.png', cv_img)
            self.latest_b64_image = base64.b64encode(buffer).decode('utf-8')
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
        if 'rail_j1' in msg.name:
            try:
                idx = msg.name.index('rail_j1')
                self.current_rail_position = msg.position[idx]
                
                # Lock in the origin on first boot
                if self.initial_rail_position is None:
                    self.initial_rail_position = self.current_rail_position
                    self.get_logger().info(f"[CALIBRATION] Set Laptop Origin to {self.initial_rail_position}m")
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
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['rail_j1', 'panda_joint6']
        msg.position = [float(rail_val), float(j6_val)]
        msg.velocity = [float(rail_speed), float(j6_speed)]
        self.rail_publisher.publish(msg)
        self.get_logger().info(f"Published JointState for rail_j1={rail_val}m and panda_joint6={j6_val}rad")

    def send_panda_joint1_command(self, target_rad: float):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['panda_joint1']
        msg.position = [float(target_rad)]
        self.rail_publisher.publish(msg)
        self.get_logger().info(f"Published JointState for panda_joint1: {target_rad}rad")

    def send_panda_search_posture(self, j1: float, j2: float, j3: float, j4: float, j5: float, j6: float, j7: float):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        msg.position = [float(j1), float(j2), float(j3), float(j4), float(j5), float(j6), float(j7)]
        self.rail_publisher.publish(msg)
        self.get_logger().info("Published JointState for full arm search posture.")

    def send_panda_joint6_command(self, target_rad: float, speed: float = None):
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
        # The background thread is updating node values automatically!
        # Just check the variables and sleep.
        current_val = node.current_rail_position if joint_name == 'rail_j1' else node.current_panda_joint1
        
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
