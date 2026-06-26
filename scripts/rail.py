import argparse
import os
from pathlib import Path
from omni.isaac.kit import SimulationApp
from arena.config import DEFAULT_CONFIG, RAIL_SCENE_PATH, RAIL_F1_WRIST_CAM_PRIM_PATH, RAIL_F1_SIDE_CAM_PRIM_PATH

# 0. Handle CLI arguments
parser = argparse.ArgumentParser(description="Arena Simulation Script")
parser.add_argument("--scene", type=str, default=RAIL_SCENE_PATH, help="Path to the USD scene file")
parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode")
parser.add_argument("--vla", type=str, default=None, help="Name of the VLA to use (e.g. 'openvla')")
args, unknown = parser.parse_known_args()

# 1. Start SimulationApp first! (Must happen before other omni imports)
launch_config = DEFAULT_CONFIG.copy()
if args.headless:
    launch_config["headless"] = True

simulation_app = SimulationApp(launch_config=launch_config)


# 2. Now it's safe to import the rest of the modules
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from pxr import Usd, UsdGeom, Gf

def set_pose(prim, pose=None, quat=None):
    xform = UsdGeom.Xformable(prim)

    translate_op = xform.GetTranslateOp()
    orient_op = xform.GetOrientOp()

    if pose is not None:
        translate_op.Set(Gf.Vec3d(*pose))

    if quat is not None:
        orient_op.Set(Gf.Quatf(*quat))


# 4. Load the Stage and World
current_scene = args.scene
print("="*60)
print(f"Loading scene: {current_scene}")
print("="*60)

open_stage(current_scene)
stage = Usd.Stage.Open(current_scene)
world = World(stage_units_in_meters=1.0)

print("="*60)
print("Scene loaded. Starting simulation...")
print("="*60)
world.reset()

wrist_cam_prim = UsdGeom.Xformable(stage.GetPrimAtPath(str(Path(RAIL_F1_WRIST_CAM_PRIM_PATH).parents[1])))
side_cam_prim = UsdGeom.Xformable(stage.GetPrimAtPath(str(Path(RAIL_F1_SIDE_CAM_PRIM_PATH).parents[1])))

# Setup ZeroMQ REP socket for VLA stage synchronization
zmq_context = None
rep_socket = None
if args.vla is not None and args.vla.lower() != "none":
    try:
        import zmq
        zmq_context = zmq.Context()
        rep_socket = zmq_context.socket(zmq.REP)
        rep_socket.bind("tcp://127.0.0.1:5555")
        print("[SYSTEM] ZMQ stage synchronization server bound to tcp://127.0.0.1:5555")
    except Exception as e:
        print(f"[SYSTEM] Warning: failed to start ZMQ REP socket in rail.py: {e}")

# 5. Main Simulation Loop
try:
    while simulation_app.is_running():
        # Step the physics and renderer
        world.step(render=True)

        if args.vla is not None and args.vla.lower() != "none" and rep_socket is not None:
            try:
                # Check for message non-blockingly
                msg = rep_socket.recv_string(flags=zmq.NOBLOCK)
                current_state = msg.lower()
                rep_socket.send_string("ok") # Acknowledge immediately

                if current_state == "pick":
                    # apple
                    print("[SYSTEM] ZMQ Pick stage detected. Moving side camera to apple pose...")
                    set_pose(side_cam_prim, pose=[0.76, -1.31, 1.36], quat=[0.68301, -0.18301, 0.18301, 0.68301])
                elif current_state == "place":
                    # bowl
                    print("[SYSTEM] ZMQ Place stage detected. Moving side camera to bowl pose...")
                    set_pose(side_cam_prim, pose=[-0.08, 1.63, 1.36], quat=[0.68301, 0.18301, 0.18301, -0.68301])

                print(f"[SYSTEM] VLA mode active ({args.vla}). Setting wrist camera pose...")
                set_pose(wrist_cam_prim, quat=[0.0, 0.5, 0.0, 0.86603])

            except zmq.Again:
                pass

except Exception as e:
    print(f"[ERROR] {e}")

# 6. Clean Shutdown
print("\nShutting down Isaac Sim...")
if rep_socket is not None:
    rep_socket.close()
if zmq_context is not None:
    zmq_context.term()
simulation_app.close()