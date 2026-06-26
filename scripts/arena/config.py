import math
import os

# --- SIMULATION CONFIGURATION ---
DEFAULT_CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RaytracedLighting"}
MAIN_SCENE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'isaac_worlds', 'Arena_wall_upd.usd'))
RAIL_SCENE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'isaac_worlds', 'rail_only.usd'))


RAIL_F1_WRIST_CAM_PRIM_PATH = "/Island/Rail_System_02/Rail_System/Franka1/panda_hand/franka1_wrist_cam/RSD455/Camera_OmniVision_OV9782_Color"
RAIL_F1_SIDE_CAM_PRIM_PATH = "/Island/Rail_System_02/Rail_System/franka1_front_cam/RSD455/Camera_OmniVision_OV9782_Color"


# --- DEVICE CONFIGURATION ---
# Each device entry defines its ROS topic, joint name(s), position limits, unit, and category.
DEVICES = {
    # --- WHITE GOODS ---
    'fridge': {
        'topic': '/sim/fridge/joint_command', 
        'joint': 'fridge_joint', 
        'min': math.radians(-170),
        'max': 0.0,
        'unit': 'degrees',
        'category': 'White Goods'
    },
    'dishwasher': {
        'topic': '/sim/dishwasher/joint_command', 
        'joint': 'dishwasher_joint', 
        'min': 0.0,
        'max': math.radians(85),
        'unit': 'degrees',
        'category': 'White Goods'
    },
    'washing_machine': {
        'topic': '/sim/washing_machine/joint_command', 
        'joint': 'washing_machine_joint', 
        'min': 0.0,
        'max': math.radians(170),
        'unit': 'degrees',
        'category': 'White Goods'
    },
    'oven': {
        'topic': '/sim/oven/joint_command', 
        'joint': 'oven_joint', 
        'min': 0.0,
        'max': math.radians(85),
        'unit': 'degrees',
        'category': 'White Goods'
    },
    
    # --- RAILS ---
    'franka_rail1': {
        'topic': '/sim/rail_franka1/joint_command', 
        'joint': 'rail_j1', 
        'min': -1.5,
        'max': 2.5,
        'unit': 'meters',
        'category': 'Rails'
    },
    'franka_rail2': {
        'topic': '/sim/rail_franka2/joint_command', 
        'joint': 'rail_j2', 
        'min': -2.9,
        'max': 0.9,
        'unit': 'meters',
        'category': 'Rails'
    }
}

# --- TABLES ---
for i in range(8):
    DEVICES[f't{i}'] = {
        'topic': f'/sim/t{i}/joint_command',
        'joint': ['j1', 'j2'], 
        'min': 0.0,
        'max': 0.65,
        'unit': 'meters',
        'category': 'Tables'
    }

# --- CAMERA TOPICS ---
CAMERA_TOPICS = [
    "/sim/rail_franka1/cam/front/color/image_raw",
    "/sim/rail_franka1/cam/wrist/color/image_raw",
    "/sim/rail_franka2/cam/front/color/image_raw",
    "/sim/rail_franka2/cam/wrist/color/image_raw",
    "/sim/wall_franka/cam/front/color/image_raw",
    "/sim/wall_franka/cam/wrist/color/image_raw",
    "/sim/wall_camera1/rgb",
    "/sim/wall_camera2/rgb"
]
