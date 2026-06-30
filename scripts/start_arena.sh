#!/bin/bash
set -e

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib

ISAACPY_PATH="$HOME/isaac-sim/python.sh"
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