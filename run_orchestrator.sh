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

RRT_MODEL_ROOT="$HOME_ROBOTICS_ROOT/src/motion_planners/data/panda"
if [ ! -f "$RRT_MODEL_ROOT/panda.urdf" ] || [ ! -f "$RRT_MODEL_ROOT/panda.srdf" ]; then
    echo "[!] Error: Panda RRT model files not found under: $RRT_MODEL_ROOT"
    exit 1
fi

echo "[+] Starting Bridge Node in background..."
python3 scripts/rail_bridge.py --ros-args \
    -p rail_damping_time_constant_sec:="${ARENA_RAIL_DAMPING_TAU_SEC:-0.18}" \
    -p rail_damping_max_speed_mps:="${ARENA_RAIL_DAMPING_MAX_SPEED_MPS:-0.35}" \
    -p rail_command_timeout_sec:="${ARENA_RAIL_COMMAND_TIMEOUT_SEC:-30.0}" &
BRIDGE_PID=$!

echo "[+] Starting RRT motion planner in background..."
rrt_supervisor() {
    local planner_pid
    local planner_status
    trap 'kill "$planner_pid" 2>/dev/null || true; wait "$planner_pid" 2>/dev/null || true; exit 0' TERM INT
    while true; do
        PYTHONFAULTHANDLER=1 python3 agent_orchestrator/src/rrt_planner.py --ros-args \
            -p planning_time_sec:=8.0 \
            -p goal_position_tolerance_m:=0.005 \
            -p goal_orientation_tolerance_rad:=0.05 \
            -p singularity_stop:="${ARENA_RRT_SINGULARITY_STOP:-0.045}" \
            -p eef_min_z_m:="${ARENA_RRT_EEF_MIN_Z_M:-0.08}" \
            -p model_root:="$RRT_MODEL_ROOT" \
            -p joint_states_topic:=/joint_states \
            -p trajectory_topic:=/rrt/tagged_joint_trajectory \
            -p controller_execution_status_topic:=/panda/controller_execution_status \
            -p controller_cancel_topic:=/panda/cancel_execution \
            -p trajectory_complete_topic:=/panda/trajectory_complete \
            -p hold_topic:=/rrt/hold_command &
        planner_pid=$!
        if wait "$planner_pid"; then
            planner_status=0
        else
            planner_status=$?
        fi
        echo "[!] RRT planner exited with status $planner_status; restarting in 1 second..."
        sleep 1
    done
}
rrt_supervisor &
RRT_PID=$!

echo "[+] Starting Cartesian Panda controller in background..."
python3 "$PANDA_CONTROLLER" --ros-args \
    -p joint_trajectory_topic:=/rrt/tagged_joint_trajectory \
    -p joint_state_topic:=/joint_states \
    -p joint_command_topic:=/cartesian/joint_command \
    -p trajectory_complete_topic:=/panda/trajectory_complete \
    -p controller_execution_status_topic:=/panda/controller_execution_status \
    -p cancel_execution_topic:=/panda/cancel_execution \
    -p controller_ready_topic:=/panda/controller_ready \
    -p vertical_action_timeout_sec:="${ARENA_VERTICAL_ACTION_TIMEOUT_SEC:-20.0}" \
    -p pose_controller_max_lin_vel:="${ARENA_VERTICAL_MAX_LINEAR_VELOCITY_MPS:-0.18}" \
    -p pose_controller_jacobian_damping:="${ARENA_VERTICAL_JACOBIAN_DAMPING:-0.035}" \
    -p pose_controller_max_joint_velocity:="${ARENA_VERTICAL_MAX_JOINT_VELOCITY_RADPS:-1.0}" >> agent_ros.log 2>&1 &
CARTESIAN_PID=$!

cleanup() {
    echo "[+] Stopping bridge, RRT planner, and Cartesian controller..."
    kill "$BRIDGE_PID" "$RRT_PID" "$CARTESIAN_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[+] Starting Robotic Task Orchestrator..."
python3 agent_orchestrator/src/main.py "$@"
