import time
import math
import rclpy
from ros_interface import wait_for_joint_target, wait_for_grasp, wait_for_place, get_shared_node

import urllib.request
import urllib.parse
import json
import subprocess


# ── TOOL EXECUTION WRAPPERS ──

def start_joint_controller() -> str:
    """Agentic Tool: Starts the global ROS 2 joint controller in the background."""
    try:
        check = subprocess.run(['tmux', 'has-session', '-t', 'global_joint_controller'], capture_output=True)
        if check.returncode == 0:
            return "Global joint controller is already running in the background."

        print("[SYSTEM]: Launching global joint controller in tmux...")
        
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'global_joint_controller'], check=True)
        
        cmd = (
            "source /opt/ros/humble/setup.bash && "
            "cd ~/classical-pipeline/panda-controller-ws/ && "
            "source install/setup.bash && "
            "ros2 run panda_python_controllers mono_controller_rail_sim --mode joint_position"
        )
        subprocess.run(['tmux', 'send-keys', '-t', 'global_joint_controller', cmd, 'C-m'], check=True)
        
        time.sleep(3.0)
        return "Success: Global joint controller started. The robot is ready to receive movement commands."
        
    except Exception as e:
        return f"Error starting controller: {e}"
        
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
            
        clean_target = target_object.lower().strip()
        
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
        
        if clean_target in ['laptop', 'home', 'start']:
            # If homing, just report success. DO NOT tell the LLM to turn the arm.
            return f"{move_result} Successfully returned to {clean_target}. The arm is safely at 0.0 rad."
        else:
            # If navigating to an object, provide the required turn instruction.
            recommended_angle = 1.57 if target_y < -0.8 else -1.57
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


def _execute_hardware_script(action: str, script_name: str, tmux_session: str, flag_attr: str, wait_func, target_object: str) -> str:
    """Core logic runner for any physical hardware bash script."""
    script_path = f"/home/homerobotics/classical-pipeline/panda-controller-ws/src/bringup/{script_name}"
    node = get_shared_node()
    
    # 1. Reset the specific completion flag dynamically (e.g., node.is_grasped = False)
    setattr(node, flag_attr, False)
    
    print(f"[EXECUTION]: Triggering classical {action} sequence: {script_path}")
    
    process = subprocess.Popen(
        ['bash', script_path, target_object], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    def cleanup_process():
        """Kills the background bash script and ensures no ghost tmux sessions remain."""
        if process.poll() is None:
            print(f"[CLEANUP]: Terminating the background {action} bash script...")
            process.terminate() 
            
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                print("[CLEANUP]: Script ignored termination. Forcing SIGKILL...")
                process.kill()
                
        # Failsafe: Kill the specific tmux session
        subprocess.run(['tmux', 'kill-session', '-t', tmux_session], capture_output=True)

    try:
        print(f"[WAITING]: Waiting for robot to call '/vlm_{action}_completed' service...")
        
        # 3. Wait using the dynamically passed function (wait_for_grasp or wait_for_place)
        if wait_func(node, timeout=60.0):
            cleanup_process()
            past_tense = "grasped" if action == "pick" else "placed"
            return f"Success: Object physically {past_tense} (Confirmed via Robot Service Call)."
            
        # 4. Handle failure cases
        retcode = process.poll()
        if retcode is not None:
            _, stderr = process.communicate()
            cleanup_process()
            if retcode == 0:
                return f"Warning: The {action} script finished cleanly, but the robot NEVER called the completion service."
            else:
                return f"Error executing {action} script (Code {retcode}). Stderr: {stderr[-500:]}"

        cleanup_process()
        return f"Error: Robot failed to complete {action} within 60 seconds (Service call timeout)."
        
    except Exception as e:
        cleanup_process()
        return f"Error triggering script: {e}"


# ── AGENTIC TOOLS ──

def execute_pick_script(target_object: str) -> str:
    """Agentic Tool: Executes classical pick script."""
    return _execute_hardware_script(
        action="pick",
        script_name="rail_demo_pick.sh",
        tmux_session="rail_demo_pick",
        flag_attr="is_grasped",
        wait_func=wait_for_grasp,
        target_object=target_object
    )

def execute_place_script(target_object: str) -> str:
    """Agentic Tool: Executes classical place script."""
    return _execute_hardware_script(
        action="place",
        script_name="rail_demo_place.sh",
        tmux_session="rail_demo_place", 
        flag_attr="is_placed",
        wait_func=wait_for_place,
        target_object=target_object
    )