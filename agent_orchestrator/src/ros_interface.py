import base64
import time
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
from std_msgs.msg import String

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
        self.rail_subscriber = self.create_subscription(JointState, '/sim/rail_franka1/joint_states', self.rail_state_callback, 10)
        self.rail_publisher = self.create_publisher(JointState, '/joint_position_command', 10)

        self.grasp_srv = self.create_service(Trigger, '/vlm_grasp_completed', self.grasp_callback)
        self.place_srv = self.create_service(Trigger, '/vlm_place_completed', self.place_callback)
        self.is_grasped = False
        self.is_placed = False
        
        self.bridge = CvBridge()
        self.latest_b64_image = None
        self.current_rail_position = None
        self.current_panda_joint1 = None
        self.initial_rail_position = None

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

    def send_absolute_rail_command(self, absolute_value: float):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['rail_j1']
        msg.position = [float(absolute_value)]
        self.rail_publisher.publish(msg)
        self.get_logger().info(f"Published Absolute JointState for rail_j1: {absolute_value}m")

    def send_panda_joint1_command(self, target_rad: float):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['panda_joint1']
        msg.position = [float(target_rad)]
        self.rail_publisher.publish(msg)
        self.get_logger().info(f"Published JointState for panda_joint1: {target_rad}rad")

def wait_for_joint_target(node: RobotHardwareInterface, joint_name: str, target_value: float, tolerance=0.02, timeout=10.0) -> bool:
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
