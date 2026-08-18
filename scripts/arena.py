import argparse
import os
from omni.isaac.kit import SimulationApp
from arena.config import DEFAULT_CONFIG, MAIN_SCENE_PATH

# 0. Handle CLI arguments
parser = argparse.ArgumentParser(description="Arena Simulation Script")
parser.add_argument("--scene", type=str, default=MAIN_SCENE_PATH, help="Path to the USD scene file")
parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode")
args, unknown = parser.parse_known_args()

# 1. Start SimulationApp first! (Must happen before other omni imports)
launch_config = DEFAULT_CONFIG.copy()
if args.headless:
    launch_config["headless"] = True

simulation_app = SimulationApp(launch_config=launch_config)

# 2. Now it's safe to import the rest of the modules
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from pxr import Usd
import rclpy

from arena.streamer import TVStreamerNode
from arena.sim_time import configure_ros2_sim_time

# 3. Initialize ROS 2 and Node
if not rclpy.ok():
    rclpy.init()
ros_node = TVStreamerNode()

# 4. Load the Stage and World
current_scene = args.scene
print("="*60)
print(f"Loading scene: {current_scene}")
print("="*60)

open_stage(current_scene)
stage = Usd.Stage.Open(current_scene)
world = World(stage_units_in_meters=1.0)
camera_nodes = configure_ros2_sim_time(stage)
print(f"ROS 2 simulation clock enabled; camera helpers switched to sim time: {camera_nodes}")

print("="*60)
print("Scene loaded. Starting simulation...")
print("="*60)
world.reset()

# 5. Main Simulation Loop
try:
    while simulation_app.is_running():
        # Step the physics and renderer
        world.step(render=True)
        
        # Process ROS 2 callbacks (gets the command or the video frame)
        rclpy.spin_once(ros_node, timeout_sec=0.0)
        
        # Update the TV screen texture if a new frame arrived
        ros_node.update_texture()

except Exception as e:
    print(f"[ERROR] {e}")

# 6. Clean Shutdown
print("\nShutting down ROS and Isaac Sim...")
ros_node.destroy_node()
if rclpy.ok():
    rclpy.shutdown()
simulation_app.close()
