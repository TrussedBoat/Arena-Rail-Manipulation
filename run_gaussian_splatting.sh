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

echo "Starting Gaussian Splatting Node..."
python3 -m gaussian_splatting_ros.node --ros-args -p use_sim_time:="$USE_SIM_TIME" "$@"
