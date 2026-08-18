# 🤖 YOLO + VLM Robotic Task Orchestrator

A hybrid ROS 2 manipulation pipeline in which a compact local VLM plans high-level actions while deterministic YOLO and ROS code perform visual search, centering, and localization.

<div align="center">
  <img src="logo.png" width="500" alt="Agent Logo">
</div>

## 🧰 Tech Stack

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E?style=for-the-badge&logo=ros)](https://docs.ros.org/en/humble/index.html)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-Vision-black?style=for-the-badge)](https://github.com/ggerganov/llama.cpp)
[![uv](https://img.shields.io/badge/uv-Fast_Python_Manager-purple?style=for-the-badge&logo=astral)](https://docs.astral.sh/uv/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](#)

## 🚀 Features

- **Compact local planner**: Runs `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` with the matching Q8_0 multimodal projector through `llama-server`.
- **Deterministic visual search**: Uses the configured local `yolo11s.pt` checkpoint; camera frames are processed locally and are not sent to the VLM during active search.
- **RRT desk scanning**: Targeted search sweeps configurable inward arcs over both desk rows while YOLO runs at a deadline-checked frame rate. Full mapping visits the minimum, centre, and maximum rail stations and executes 11 semantic-mapping viewpoints per desk side.
- **Close confirmation**: Holds RRT motion on a candidate, depth-localizes it, approaches its rail X, and persists only a strict closer-look detection.
- **Dynamic coordinates**: Saves rail-zero global XYZ pickup coordinates to `semantic_distances_dynamic.json` without overwriting unrelated labels.
- **Fail-closed orchestration**: A LangGraph stage gate permits only one expected high-level tool at a time and terminates immediately after search, pick, navigation, or place failure.

## Architecture

### RRT end-effector commands

The orchestrator exposes `move_eef_to_pose` with XYZ in metres and roll, pitch,
and yaw in radians. It publishes `geometry_msgs/Pose` on `/rrt/pose_command`, expressed
relative to `panda_link0`. The RRT planner converts the pose into a timed
seven-joint trajectory, validates self-collision, joint limits, and Jacobian
singularity margin, and publishes it on `/rrt/joint_trajectory`. The caller
waits for `/panda/controller_state` and receives failures immediately from
`/rrt/status`.

The planner rejects any target or interpolated path state whose EEF height in
`panda_link0` would be below `0.08 m`. This calibrated EEF floor is the
`eef_min_z_m` ROS parameter, set by `run_orchestrator.sh`; override it with
`ARENA_RRT_EEF_MIN_Z_M` when recalibrating the gripper geometry.

The planner warns below a singularity metric of `0.10` and stops below `0.045`.
Override the stop limit with `ARENA_RRT_SINGULARITY_STOP` only when calibrating
reachable manipulation poses.

The low-level controller does not consume `/rrt/pose_command` directly. Runtime safety
monitoring issues a zero-velocity joint hold if actual feedback enters an unsafe
configuration. Direct joint commands can still take priority through the bridge;
a later `/rrt/pose_command` re-arms planned trajectory forwarding.
Targeted search may also publish `/rrt/cancel`; this creates an intentional
current-joint hold and reports `cancelled` separately from runtime safety stops.

### ROS command topics

| Topic | Producer → consumer | Purpose |
|---|---|---|
| `/direct_joint_command` | orchestrator → bridge | Direct rail/Panda joint targets. |
| `/rrt/pose_command` | orchestrator → RRT planner | Cartesian EEF target. |
| `/rrt/joint_trajectory` | RRT planner → Panda controller | Validated joint trajectory. |
| `/cartesian/joint_command` | Panda controller → bridge | Streamed joint targets while executing a trajectory. |
| `/rrt/hold_command` | RRT planner → bridge | Emergency or cancellation joint hold. |
| `/gripper/command` | orchestrator → bridge/Panda controller | Gripper command. |
| `/panda/controller_ready` | Panda controller → clients | Controller startup readiness. |
| `/panda/controller_state` | Panda controller → clients/RRT planner | Trajectory completion notification. |

The old `/joint_position_cmd` controller input is removed. Do not publish the
old `/joint_position_command`, `/joint_command`, `/pose_cmd`, or
`/joint_trajectory_cmd` names after restarting the stack.

`run_orchestrator.sh` starts the controller from the `home_robotics` workspace.
Set `ARENA_HOME_ROBOTICS_SETUP` if its `install/setup.bash` is not at the default
location. Cartesian timeouts and TF frame names can be overridden with the
`YOLO_VLM_CARTESIAN_*` environment variables defined in `src/config.py`. RRT
parameters such as planning time, edge resolution, and singularity thresholds
are ROS parameters declared by `src/rrt_planner.py`.

The VLM sees only six high-level tools:

1. `start_joint_controller`
2. `targeted_search`
3. `execute_pick_script`
4. `move_rail_to_object`
5. `execute_place_script`
6. `finish_task`

Camera capture, raw joint inspection, wrist motion, rail stepping, YOLO inference, centering, and coordinate persistence remain internal Python operations.

The enforced manipulation sequence is:

```text
initialize controller
  -> search and localize pickup target with YOLO
  -> execute existing pick script
  -> move to the fixed purple bowl
  -> execute existing place script
  -> return to home
  -> finish task
```

Every successful localization writes the object position in the rail-zero
`global_origin` frame. `rail_j1 = 0` is the fixed global origin and the axes
are rail-aligned (the robot base is rotated 180° about Z relative to them):

```json
{
  "apple": {
    "x": 1.234,
    "y": -0.120,
    "z": 0.045
  }
}
```

`run_sim.sh` starts the reusable `rail_global` reference-frame node alongside
the simulator. It consumes `/sim/rail_franka1/joint_states` and publishes
`global_origin -> panda_link0`. It fails rather than publishing if another TF
parent for `panda_link0` is detected. Existing entries for other labels are
preserved.

## Runtime configuration

Runtime settings are loaded and validated by `src/config.py` before ROS motion begins. Defaults target:

- Qwen2.5-VL-7B Q4_K_M with Q8_0 mmproj
- 4096-token context, one inference slot, 99 GPU layers, Flash Attention, and 128–256 completion tokens
- `yolo11s.pt` with a strict final confidence requirement of `> 0.85`
- targeted-search defaults of seven RRT views per desk side, 5 FPS, 0.30 candidate confidence, 1.0 m desk width, and 0.30 m close stand-off
- rail limits, motion tolerances, camera geometry, and semantic-coordinate paths

Paths can be overridden with the `YOLO_VLM_*` environment variables defined in `src/config.py`. Missing model files or scripts fail validation; the orchestrator never downloads them automatically.

## 🛠️ Installation & Setup

### 1. Install Prerequisites

#### Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# OR
wget -qO- https://astral.sh/uv/install.sh | sh
```

### 2. Setup Python Environment (using `uv`)
```bash
# Initialize uv project (if not already)
uv init --python 3.10

# Create and activate virtual environment
uv venv agent_orchestrator
source agent_orchestrator/bin/activate

# Install dependencies
uv pip install -r requirements.txt 
```

## 🎮 Running the Orchestrator for Neo-Classic Pipeline

To run the full system, open two separate terminals:

### Terminal 1: Start the Arena Simulation
```bash
cd Workspace/Arena-RealSim
./scripts/start_arena.sh rail
```

### Terminal 2: Run the Orchestrator Agent
```bash
cd Workspace/Arena-RealSim
source agent_orchestrator/agent_orchestrator/bin/activate
source /opt/ros/humble/setup.bash
python3 agent_orchestrator/src/main.py
```

## 📂 Project Structure

- **`src/config.py`**: Environment-based runtime configuration and fail-fast validation.
- **`src/main.py`**: Main entry point and interactive CLI runtime loop.
- **`src/agent.py`**: Minimal six-tool LangGraph planner and enforced pipeline stages.
- **`src/tools.py`**: llama.cpp lifecycle, deterministic YOLO search, centering, persistence, navigation, and manipulation wrappers.
- **`src/ros_interface.py`**: ROS 2 Node integration, subscription handlers, and hardware actuation interface.
