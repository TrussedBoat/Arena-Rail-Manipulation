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

# Activate the agent environment before sourcing ROS so ROS Python packages,
# including cv_bridge, remain visible inside the virtual environment.
VENV_PATH="$SCRIPT_DIR/agent_orchestrator/agent_env/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "[+] Activating agent virtual environment ($VENV_PATH)..."
    source "$VENV_PATH"
else
    echo "[!] Error: Virtual environment not found at $VENV_PATH"
    echo "    Run setup to create the environment."
    exit 1
fi

# Source ROS 2 Humble
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "[+] Sourcing ROS 2 Humble (Domain ID: $ROS_DOMAIN_ID, localhost only)..."
    source /opt/ros/humble/setup.bash
else
    echo "[!] Warning: /opt/ros/humble/setup.bash not found."
fi

HOME_ROBOTICS_SETUP="${ARENA_HOME_ROBOTICS_SETUP:-$HOME/workspace/home_robotics/homerobotics_ws/install/setup.bash}"
if [ ! -f "$HOME_ROBOTICS_SETUP" ]; then
    echo "[!] Error: home_robotics setup script not found: $HOME_ROBOTICS_SETUP"
    echo "    Build homerobotics_ws or set ARENA_HOME_ROBOTICS_SETUP."
    exit 1
fi
echo "[+] Sourcing home_robotics controller workspace..."
source "$HOME_ROBOTICS_SETUP"

# ros2 run executes the generated panda_controller launcher with its embedded
# /usr/bin/python3 shebang. Invoke that launcher as a script through the
# activated environment's python3 instead, so NumPy/SciPy/Robotics Toolbox
# resolve from agent_env consistently.
HOME_ROBOTICS_ROOT="$(dirname "$(dirname "$HOME_ROBOTICS_SETUP")")"
PANDA_CONTROLLER="$HOME_ROBOTICS_ROOT/install/controllers/lib/controllers/panda_controller"
if [ ! -f "$PANDA_CONTROLLER" ]; then
    echo "[!] Error: panda_controller launcher not found: $PANDA_CONTROLLER"
    exit 1
fi

echo "[+] Starting Bridge Node in background..."
python3 scripts/rail_bridge.py &
BRIDGE_PID=$!

echo "[+] Starting Cartesian Panda controller in background..."
python3 "$PANDA_CONTROLLER" --ros-args \
    -p cartesian_pose_topic:=/pose_cmd \
    -p joint_state_topic:=/joint_states \
    -p joint_command_topic:=/joint_command \
    -p controller_state_topic:=/controller_state \
    -p controller_ready_topic:=/controller_ready &
CARTESIAN_PID=$!

cleanup() {
    echo "[+] Stopping bridge and Cartesian controller..."
    kill "$BRIDGE_PID" "$CARTESIAN_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[+] Starting Robotic Task Orchestrator..."
python3 agent_orchestrator/src/main.py
