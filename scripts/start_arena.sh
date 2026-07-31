#!/bin/bash
set -e

# Detect Isaac Sim installation directory (~/isaacsim or ~/isaac-sim)
ISAAC_DIR="$HOME/isaacsim"
if [ ! -d "$ISAAC_DIR" ] && [ -d "$HOME/isaac-sim" ]; then
    ISAAC_DIR="$HOME/isaac-sim"
fi

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$ISAAC_DIR/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ISAAC_DIR/exts/isaacsim.ros2.bridge/humble/rclpy:$PYTHONPATH

ISAACPY_PATH="$ISAAC_DIR/python.sh"
ARENA_PY_PATH="$(dirname "$0")/arena.py"

while [ $# -gt 0 ]; do
    case "$1" in
        rail)
            ARENA_PY_PATH="$(dirname "$0")/rail.py"
            shift
            ;;
        *)
            if [[ "$1" == *.sh ]] || [[ "$1" == *"python.sh"* ]]; then
                ISAACPY_PATH="$1"
            fi
            shift
            ;;
    esac
done

exec "$ISAACPY_PATH" "$ARENA_PY_PATH" --enable isaacsim.ros2.bridge