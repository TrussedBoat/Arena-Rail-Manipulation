import os
import json
import numpy as np
import cv2

# ===================================
# Starting the simulation (first thing first)
# ===================================
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

# ===================================
# Importing the libraries
# ===================================
import omni.usd
from pxr import UsdGeom, Gf, Usd
import omni.replicator.core as rep
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core import World


# ===================================
# HELPER METHODS
# ===================================

def get_intrinsic_matrix_from_usd_camera(camera_prim: UsdGeom.Camera, width: int, height: int):
    focal_length = camera_prim.GetFocalLengthAttr().Get()
    horiz_aperture = camera_prim.GetHorizontalApertureAttr().Get()
    vert_aperture = camera_prim.GetVerticalApertureAttr().Get()
    
    horiz_offset = camera_prim.GetHorizontalApertureOffsetAttr().Get() or 0.0
    vert_offset = camera_prim.GetVerticalApertureOffsetAttr().Get() or 0.0

    fx = width * focal_length / horiz_aperture
    fy = height * focal_length / vert_aperture
    cx = width / 2.0
    cy = height / 2.0

    if horiz_offset != 0.0:
        cx -= (horiz_offset / horiz_aperture) * width
    if vert_offset != 0.0:
        cy += (vert_offset / vert_aperture) * height

    return [
        [fx,  0.0, cx ],
        [0.0, fy,  cy ],
        [0.0, 0.0, 1.0]
    ]

def save_replica_intrinsics(camera_matrix, width, height, output_dir):
    fx, cx = camera_matrix[0][0], camera_matrix[0][2]
    fy, cy = camera_matrix[1][1], camera_matrix[1][2]
    
    camera_params = {
        "camera": {
            "w": width, "h": height,
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "scale": 6553.5
        }
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "camera_params.json"), "w") as f:
        json.dump(camera_params, f, indent=4)

def get_replica_formatted_pose(prim):
    x = UsdGeom.Xformable(prim)
    mat = x.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    # 1. Get standard numpy matrix (transpose to row-major)
    usd_pose_matrix = np.array(mat, dtype=np.float64).T 

    # 2. Mathematically strip the scale from the top-left 3x3 rotation matrix
    for i in range(3):
        # Calculate the length of the vector
        norm = np.linalg.norm(usd_pose_matrix[i, :3])
        if norm > 1e-6:
            # Divide the vector by its length to normalize it to 1.0
            usd_pose_matrix[i, :3] = usd_pose_matrix[i, :3] / norm

    # 3. Apply OpenCV (Z-forward, Y-down) conversion
    R_usd_to_cv = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)
    
    cv_pose_matrix = usd_pose_matrix @ R_usd_to_cv
    
    return " ".join([f"{val:.18e}" for val in cv_pose_matrix.flatten()])

def move_camera(camera_prim_path, frame_idx, total_frames, start_pos, start_rot):
    """
    Moves the camera along a rectangular path:
    Right (+1.6m), Down (+0.35m), Left (-1.6m), Up (-0.35m)
    """
    camera_xform = XFormPrim(prim_path=camera_prim_path)
    start_x, start_y, start_z = start_pos
    
    # Perimeter segment lengths
    len_right = 1.6
    len_down  = 0.35
    len_left  = 1.6
    len_up    = 0.35
    
    total_distance = len_right + len_down + len_left + len_up # 3.9m total
    
    # Calculate progress along the total path (0.0 to 1.0)
    progress = frame_idx / max(1, (total_frames - 1))
    current_dist = progress * total_distance
    
    rel_x, rel_y = 0.0, 0.0
    
    if current_dist <= len_right:
        rel_x = current_dist
        rel_y = 0.0
    elif current_dist <= (len_right + len_down):
        rel_x = len_right
        rel_y = current_dist - len_right
    elif current_dist <= (len_right + len_down + len_left):
        rel_x = len_right - (current_dist - len_right - len_down)
        rel_y = len_down
    else:
        rel_x = 0.0
        rel_y = len_down - (current_dist - len_right - len_down - len_left)

    new_pos = np.array([start_x + rel_x, start_y + rel_y, start_z])
    
    # Apply the world pose (keeping the original rotation)
    camera_xform.set_world_pose(position=new_pos, orientation=start_rot)


# ===================================
# MAIN FUNCTION
# ===================================
def main():
    # Fetch environment variables
    OUTPUT_DIR = os.environ.get("DATASET_OUTPUT_DIR", "/home/homerobotics/Workspace/Arena-RealSim/scripts/mapping/dataset")
    CAMERA_PRIM_PATH = os.environ.get("CAMERA_PRIM_PATH", "/Island/Rail_System_02/Rail_System/franka2_front_cam/RSD455/Camera_OmniVision_OV9782_Color")
    IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", "1280"))
    IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", "720"))
    NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "100"))
    USD_SCENE_PATH = os.environ.get("USD_SCENE_PATH", "/home/homerobotics/Workspace/Arena-RealSim/isaac_worlds/rail_only.usd")

    if not USD_SCENE_PATH or not os.path.exists(USD_SCENE_PATH):
        print(f"ERROR: Could not find USD file at: {USD_SCENE_PATH}")
        simulation_app.close()
        return

    print(f"Loading scene: {USD_SCENE_PATH}")
    omni.usd.get_context().open_stage(USD_SCENE_PATH)
    
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
        
    print("="*60)
    print("Scene loaded. Initializing World...")
    print("="*60)

    # Initialize World API matching benchmark logic
    world = World(stage_units_in_meters=1.0)
    world.reset()

    color_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(color_dir, exist_ok=True)
    traj_filepath = os.path.join(OUTPUT_DIR, "traj.txt")
    if os.path.exists(traj_filepath):
        os.remove(traj_filepath)

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    
    # Save Intrinsics
    usd_camera = UsdGeom.Camera(camera_prim)
    intrinsic_matrix = get_intrinsic_matrix_from_usd_camera(usd_camera, IMAGE_WIDTH, IMAGE_HEIGHT)
    save_replica_intrinsics(intrinsic_matrix, IMAGE_WIDTH, IMAGE_HEIGHT, OUTPUT_DIR)

    # Setup Replicator Annotators
    render_product = rep.create.render_product(CAMERA_PRIM_PATH, resolution=(IMAGE_WIDTH, IMAGE_HEIGHT))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_annotator.attach(render_product)
    depth_annotator.attach(render_product)

    camera_xform = XFormPrim(prim_path=CAMERA_PRIM_PATH)
    start_pos, start_rot = camera_xform.get_world_pose()

    print(f"Starting collection of {NUM_FRAMES} frames...")
    print(f"Camera starting at: {start_pos}")

    # Allow simulation to settle using standard World steps
    for _ in range(10):
        world.step(render=True)

    for i in range(NUM_FRAMES):
        # Move Camera
        move_camera(CAMERA_PRIM_PATH, i, NUM_FRAMES, start_pos, start_rot)
        
        # Step World (Automatically ticks physics and replicator/rendering)
        world.step(render=True)
        
        # Save Images
        rgb_data = rgb_annotator.get_data()
        depth_data = depth_annotator.get_data()
        
        if rgb_data is None or depth_data is None:
            continue

        rgb_img = cv2.cvtColor(rgb_data[..., :3], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(color_dir, f"frame_{i:06d}.jpg"), rgb_img)

        scaled_depth = np.clip(depth_data * 6553.5, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(color_dir, f"depth_{i:06d}.png"), scaled_depth)

        # Save Pose
        pose_string = get_replica_formatted_pose(camera_prim)
        with open(traj_filepath, "a") as f:
            f.write(pose_string + "\n")

    print(f"Saved dataset to: {OUTPUT_DIR}")
    simulation_app.close()


if __name__ == "__main__":
    main()