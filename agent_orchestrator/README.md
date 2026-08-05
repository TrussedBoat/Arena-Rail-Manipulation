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
- **Stop-and-go motion safety**: YOLO runs only after rail and wrist convergence, a stability interval, and acquisition of a fresh camera frame.
- **Visual centering and recovery**: Applies bounded rail corrections and backtracks to the last observed position if the target temporarily disappears.
- **Dynamic coordinates**: Saves the session-relative pickup coordinate to `semantic_distances_dynamic.json` without overwriting unrelated labels.
- **Fail-closed orchestration**: A LangGraph stage gate permits only one expected high-level tool at a time and terminates immediately after search, pick, navigation, or place failure.

## Architecture

### Cartesian end-effector commands

The orchestrator exposes `move_eef_to_pose` with XYZ in metres and roll, pitch,
and yaw in radians. It publishes `geometry_msgs/Pose` on `/pose_cmd`, expressed
relative to `panda_link0`, and waits for `/controller_state` before continuing.
After the first Cartesian command, the Cartesian controller exclusively owns
Panda arm and gripper joints; rail-only commands remain available.

`run_orchestrator.sh` starts the controller from the `home_robotics` workspace.
Set `ARENA_HOME_ROBOTICS_SETUP` if its `install/setup.bash` is not at the default
location. Cartesian timeouts and TF frame names can be overridden with the
`YOLO_VLM_CARTESIAN_*` environment variables defined in `src/config.py`.

The VLM sees only six high-level tools:

1. `start_joint_controller`
2. `search_and_locate_with_yolo`
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

Every successful localization writes only the session-relative rail coordinate:

```json
{
  "apple": {
    "x": 1.234
  }
}
```

The coordinate is calculated as `current_absolute_rail_j1 - initial_rail_position`. Existing entries for other labels are preserved.

## Runtime configuration

Runtime settings are loaded and validated by `src/config.py` before ROS motion begins. Defaults target:

- Qwen2.5-VL-7B Q4_K_M with Q8_0 mmproj
- 4096-token context, one inference slot, 99 GPU layers, Flash Attention, and 128–256 completion tokens
- `yolo11s.pt` with a strict final confidence requirement of `> 0.85`
- rail limits, waypoint spacing, wrist search angles, centering gains, tolerances, and semantic-coordinate paths

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
