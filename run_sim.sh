#!/bin/bash
set -e

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source ROS 2 Humble
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "[+] Sourcing ROS 2 Humble..."
    source /opt/ros/humble/setup.bash
else
    echo "[!] Warning: /opt/ros/humble/setup.bash not found."
fi

# Detect Isaac Sim Python executable location
ISAAC_PY=""
if [ -f "$HOME/isaacsim/python.sh" ]; then
    ISAAC_PY="$HOME/isaacsim/python.sh"
elif [ -f "$HOME/isaac-sim/python.sh" ]; then
    ISAAC_PY="$HOME/isaac-sim/python.sh"
else
    echo "[!] Error: Isaac Sim python.sh not found in ~/isaacsim or ~/isaac-sim."
    exit 1
fi

echo "[+] Starting Arena RealSim Simulation with: $ISAAC_PY"
exec ./scripts/start_arena.sh rail "$ISAAC_PY"
