#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# Source the base ROS installation
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Ensure the package is in PYTHONPATH so it can be run as a module without building
export PYTHONPATH="$SCRIPT_DIR/gaussian_splatting_ros${PYTHONPATH:+:$PYTHONPATH}"

# Arena publishes /clock, so simulation time is the safe default.
USE_SIM_TIME="${ARENA_USE_SIM_TIME:-true}"
TRAINING_ITERATIONS="${ARENA_3DGS_TRAINING_ITERATIONS:-20000}"
DEPTH_SUPERVISION_ENABLED="${ARENA_3DGS_DEPTH_SUPERVISION:-true}"

echo "Starting Gaussian Splatting Node..."

# Inject CUDA and the virtual environment into the PATH so the subprocess can find ninja and nvcc!
export PATH="/usr/local/cuda/bin:/home/thinkstation-sim/workspace/Arena-RealSim/gaussian_splatting_ros/nerfstudio_env/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

python3 -m gaussian_splatting_ros.node --ros-args \
  -p use_sim_time:="$USE_SIM_TIME" \
  -p training_command:="ns-train" \
  -p training_iterations:="$TRAINING_ITERATIONS" \
  -p depth_supervision.enabled:="$DEPTH_SUPERVISION_ENABLED" \
  -p object_point_voxel_size_m:=0.01 \
  -p scene_background_voxel_size_m:=0.03 \
  "$@"
