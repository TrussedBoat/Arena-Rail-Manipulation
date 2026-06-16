#!/bin/bash

# 1. Define your dataset parameters via environment variables
export DATASET_OUTPUT_DIR="/home/homerobotics/Workspace/Arena-RealSim/scripts/mapping/dataset"
export USD_SCENE_PATH="/home/homerobotics/Workspace/Arena-RealSim/isaac_worlds/rail_only.usd"
export CAMERA_PRIM_PATH="/Island/Rail_System_02/Rail_System/franka2_front_cam/RSD455/Camera_OmniVision_OV9782_Color"
export IMAGE_WIDTH="1280"
export IMAGE_HEIGHT="720"
export NUM_FRAMES="150"

# 2. Define the path to your Isaac Sim installation
# (Update this path to wherever Isaac Sim 5.1 is installed on your Ubuntu machine)
ISAAC_SIM_DIR="${HOME}/isaac-sim"

# 3. Run the python script using Isaac Sim's bundled python environment
echo "Launching Isaac Sim Python Environment..."
${ISAAC_SIM_DIR}/python.sh construct_offline_map.py