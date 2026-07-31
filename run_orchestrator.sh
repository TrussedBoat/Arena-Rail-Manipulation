#!/bin/bash
set -e

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure local bin is in PATH for uv/python binaries
export PATH="$HOME/.local/bin:$PATH"

# ROS 2 network isolation — must match the domain used by run_sim.sh
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Source ROS 2 Humble
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "[+] Sourcing ROS 2 Humble (Domain ID: $ROS_DOMAIN_ID, localhost only)..."
    source /opt/ros/humble/setup.bash
else
    echo "[!] Warning: /opt/ros/humble/setup.bash not found."
fi


# Activate agent virtual environment
VENV_PATH="$SCRIPT_DIR/agent_orchestrator/agent_env/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "[+] Activating agent virtual environment ($VENV_PATH)..."
    source "$VENV_PATH"
else
    echo "[!] Error: Virtual environment not found at $VENV_PATH"
    echo "    Run setup to create the environment."
    exit 1
fi

echo "[+] Starting Bridge Node in background..."
python3 scripts/rail_bridge.py &
BRIDGE_PID=$!

trap "echo '[+] Stopping Bridge Node...'; kill $BRIDGE_PID" EXIT

echo "[+] Starting Robotic Task Orchestrator..."
python3 agent_orchestrator/src/main.py
