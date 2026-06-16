import time
import math
import rclpy
from ros_interface import wait_for_joint_target, get_shared_node

import urllib.request
import urllib.parse
import json


# ── TOOL EXECUTION WRAPPERS ──
def get_latest_ros_image(timeout_sec=10.0) -> str:
    node = get_shared_node()
    start = time.time()
    print("Waiting for image from ROS 2 topic...")
    while node.latest_b64_image is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (time.time() - start) > timeout_sec:
            node.destroy_node()
            raise TimeoutError("Timed out waiting for ROS 2 image message.")
    img_data = node.latest_b64_image
    return img_data

def get_current_joint_states() -> dict:
    node = get_shared_node()
    start = time.time()
    while node.current_rail_position is None or node.current_panda_joint1 is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (time.time() - start) > 5.0:
            node.destroy_node()
            raise TimeoutError("Could not fetch active joint telemetry.")
    states = {
        "rail_j1_meters": node.current_rail_position,
        "panda_joint1_radians": node.current_panda_joint1
    }
    return states

def move_rail_relative(relative_distance_m: float) -> str:
    node = get_shared_node()
    
    # Wait for initial telemetry
    start = time.time()
    while node.current_rail_position is None:
        time.sleep(0.1)
        if (time.time() - start) > 5.0:
            return "Error: Could not read current rail position."
            
    current_pos = node.current_rail_position
    absolute_target = current_pos + relative_distance_m
    
    print(f"[MATH]: Moving to {absolute_target:.4f}m...")
    node.send_absolute_rail_command(absolute_target)
    
    # Use your helper to verify
    success = wait_for_joint_target(node, 'rail_j1', absolute_target, tolerance=0.01)
    
    return f"Success: Moved to {absolute_target:.4f}m." if success else "Warning: Timeout during move."

def move_rail_to_object(target_object: str) -> str:
    """Agentic Tool: Safely homes arm, calculates absolute world target, and moves rail."""
    try:
        with open("semantic_distances.json", "r") as f:
            distances = json.load(f)
            
        clean_target = target_object.lower().strip().replace(" ", "_")
        
        # 1. Get Semantic Offsets
        if clean_target in ['laptop', 'home', 'start']:
            target_offset_x = 0.0
            target_y = 0.0 
        elif clean_target in distances:
            target_offset_x = distances[clean_target]["x"]
            target_y = distances[clean_target]["y"]
        else:
            return f"Error: '{target_object}' not found in map."
            
        node = get_shared_node()
        
        # Wait for calibration
        start_wait = time.time()
        while node.current_rail_position is None or node.initial_rail_position is None:
            time.sleep(0.1)
            if time.time() - start_wait > 5.0:
                return "Error: Hardware not calibrated. Missing initial_rail_position."

        # 2. AUTO-SAFETY: Force arm to 0.0 before moving
        if node.current_panda_joint1 is not None and abs(node.current_panda_joint1) > 0.02:
            print("[SAFETY INTERLOCK] Arm is deployed. Auto-homing to 0.0 rad before rail movement...")
            node.send_panda_joint1_command(0.0)
            wait_for_joint_target(node, 'panda_joint1', 0.0)
            
        # 3. Hardware Origin Math
        # Absolute Target = Start Location (-1.09m) + Semantic Offset (2.31m)
        absolute_target = node.initial_rail_position + target_offset_x
        relative_move = absolute_target - node.current_rail_position
        
        # 4. Actuate Rails
        move_result = move_rail_relative(relative_move)
        
        # 5. Determine Y-Axis Orientation
        recommended_angle = 1.57 if target_y < -1.0 else -1.57
            
        return (f"{move_result} The object '{target_object}' is at Y: {target_y}m. "
                f"You MUST now call 'turn_panda_arm' with target_rad={recommended_angle} to face it.")
        
    except Exception as e:
        return f"Error executing navigation: {e}"

def turn_panda_arm(target_rad: float) -> str:
    """Agentic Tool: Turns panda_joint1 to face the object."""
    node = get_shared_node()
    node.send_panda_joint1_command(target_rad)
    print(f"[WAITING]: Tracking joint_states until panda_joint1 reaches {target_rad} rad...")
    
    success = wait_for_joint_target(node, 'panda_joint1', target_rad)
    if success:
        return f"Successfully turned panda_joint1 to face the workspace ({target_rad} rad)."
    else:
        return f"Warning: Timed out waiting for panda_joint1 to reach {target_rad} rad."

def home_panda_arm() -> str:
    node = get_shared_node()
    # Ensure we have the current state first
    start = time.time()
    while node.current_panda_joint1 is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - start > 5.0:
            return "Error: Could not read panda_joint1 telemetry."
            
    target_rad = 0.0
    # Check if already home
    if abs(node.current_panda_joint1 - target_rad) <= 0.02:
        return "Panda manipulator arm is already at the home configuration (0.0 rad)."
        
    node.send_panda_joint1_command(target_rad)
    print("[WAITING]: Tracking joint_states until panda_joint1 reaches home (0.0 rad)...")
    
    success = wait_for_joint_target(node, 'panda_joint1', target_rad)
    
    if success:
        return "Panda manipulator arm has successfully returned to home default configuration (0.0 rad)."
    else:
        return "Warning: Arm homing command dispatched, but timed out verifying final position."