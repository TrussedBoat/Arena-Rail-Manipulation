#!/bin/bash
set -e

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ROS 2 network isolation — must match the domain used by run_orchestrator.sh
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
./scripts/start_arena.sh rail "$ISAAC_PY" &
SIM_PID=$!

cleanup() {
    kill "$FRAME_PID" "$SIM_PID" 2>/dev/null || true
    wait "$FRAME_PID" "$SIM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[+] Starting rail-zero global TF publisher..."
PYTHONPATH="$SCRIPT_DIR/agent_orchestrator/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 agent_orchestrator/src/rail_global.py &
FRAME_PID=$!

wait "$SIM_PID"
