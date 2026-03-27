#!/bin/bash
set -e

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib

ISAACPY_PATH=${1:-"$HOME/isaac-sim/python.sh"}
ARENA_PY_PATH="$(dirname "$0")/arena.py"

exec "$ISAACPY_PATH" "$ARENA_PY_PATH" --enable isaacsim.ros2.bridge
