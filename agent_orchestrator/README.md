# 🤖 Orchestrator: VLM agent + RRT planner

The high-level brain of Arena RealSim. A local vision-language model (VLM) reads a plain-language task and calls a small set of robot tools. The tools search, navigate, pick and place using semantic perception, an RRT motion planner and the rail bridge.

For the whole system (ROS message flow, motion layers, perception, splatting), see the [root README](../README.md).

<div align="center">
  <img src="logo.png" width="500" alt="Agent Logo">
</div>

## 🧰 Tech stack

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E?style=for-the-badge&logo=ros)](https://docs.ros.org/en/humble/index.html)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?style=for-the-badge)](https://www.langchain.com/langgraph)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-Vision-black?style=for-the-badge)](https://github.com/ggerganov/llama.cpp)
[![uv](https://img.shields.io/badge/uv-Fast_Python_Manager-purple?style=for-the-badge&logo=astral)](https://docs.astral.sh/uv/)

## 🚀 Run

From the repository root, with the simulator (`./run_sim.sh`) and perception (`./run_semantic_perception.sh`) already running:

```bash
./run_orchestrator.sh              # add --show-think to print the VLM's reasoning
```

The script starts, in order:

1. `scripts/rail_bridge.py`: the only node that commands Isaac Sim
2. `rrt_planner.py`: restarted automatically if it exits
3. the Panda controller from the `home_robotics` workspace (log: `agent_ros.log`)
4. `src/main.py`: starts `llama-server` in a tmux session (`vlm_server`), then opens an interactive prompt

Type a task at the prompt (an empty line runs the default *"pick up the apple and place it into the purple bowl"*). `quit` stops everything.

## 🛠️ Setup

You need Ubuntu 22.04 with ROS 2 Humble, `tmux`, a CUDA GPU, and a built `home_robotics` workspace (custom `interface` messages, the Panda controller and the Panda URDF/SRDF).

```bash
# from the repository root
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv agent_orchestrator/agent_env --python 3.10
source agent_orchestrator/agent_env/bin/activate
uv pip install -r agent_orchestrator/requirements.txt
```

Other things you must provide (nothing is downloaded automatically):

| What | Default location | Override |
|---|---|---|
| `llama-server` build | `<llama root>/build/bin/llama-server` | `YOLO_VLM_LLAMA_ROOT`, `YOLO_VLM_LLAMA_EXECUTABLE` |
| VLM: Qwen3-VL-8B-Instruct Q4_K_M + F16 mmproj (GGUF) | `<llama root>/models/` | `YOLO_VLM_MODEL_PATH`, `YOLO_VLM_MMPROJ_PATH` |
| YOLO weights | `agent_orchestrator/models/yolo26x.pt` | `YOLO_VLM_YOLO_CHECKPOINT` |
| `home_robotics` workspace | `<controller root>` | `YOLO_VLM_CONTROLLER_WORKSPACE`, `ARENA_HOME_ROBOTICS_SETUP` |

> ⚠️ The default `<llama root>` and `<controller root>` in `src/config.py` point to the development machine's home directory. On any other machine, set the overrides above.

## ⚙️ Configuration

All settings are loaded and checked in `src/config.py` before any robot motion. They come from `YOLO_VLM_*` environment variables (about 100, all listed in that file: rail limits, scan geometry, pick/place offsets, timeouts, camera intrinsics). If a check fails, the program stops with a clear error.

Checked at startup:
- required files exist: `llama-server`, GGUF, mmproj, YOLO checkpoint, `semantic_distances.json` (static coordinates), the ROS and controller setup scripts
- the VLM fits its GPU budget, which is fixed to 4096-token context, 1 slot, 99 GPU layers, Flash Attention, 128–256 completion tokens, at most 8 GB VRAM
- the YOLO checkpoint knows the required classes (`apple` by default)

The old `rail_demo_pick.sh` / `rail_demo_place.sh` scripts are no longer used by the current tools. A missing script only prints a warning (`YOLO_VLM_ALLOW_MOCK_HARDWARE_SCRIPTS` defaults to on).

## 🧠 How the agent works

`agent.py` builds a **LangGraph** loop: `call_llm` → `execute_tools` → `call_llm` … until `finish_task`, a failure, or 20 steps. Every tool call must carry a `thought` field.

| Area | Tools |
|---|---|
| Setup / end | `start_joint_controller`, `finish_task` |
| Semantic map | `search_semantic_objects`, `targeted_search`, `general_mapping` |
| Navigation | `move_rail_to_object`, `move_rail_relative`, `scan_object` |
| Arm | `turn_panda_arm`, `home_panda_arm` |
| Manipulation | `get_camera_frame`, `execute_pick_script`, `execute_place_script` |

Rules enforced in code, not only in the prompt:
- the first call must be `start_joint_controller`
- `scan_object` is allowed only right after `move_rail_to_object` on the same object ID
- pick and place are refused unless the previous call was `get_camera_frame` and its separate, context-free VLM check returned `target_confirmed` and `safe_for_action`
- a failed pick or place has already tried a safe retreat, so the failure is recoverable and the VLM re-plans

Objects are chosen by exact `object_id` from `semantic_objects.json`, the map written by semantic perception. Positions are in the rail-zero `global_origin` frame.

## 📂 Files

| File | Purpose |
|---|---|
| `src/main.py` | Entry point, system prompt, interactive loop, cleanup of tmux sessions |
| `src/agent.py` | Tool schemas, stage gate, LangGraph graph |
| `src/tools.py` | The tools: search, mapping, navigation, pick/place, VLM server and visual verification |
| `src/ros_interface.py` | Shared ROS node: action/service clients, camera, TF, joint state |
| `src/rrt_planner.py` | RRT planning with joint-limit, collision, height and singularity checks |
| `src/rail_global.py` | Publishes `global_origin → panda_link0` |
| `src/targeted_scan_geometry.py` | Scan-arc and look-at pose maths |
| `src/config.py` | Environment-driven configuration and validation |
| `src/ros_logger.py` | Sends ROS node logs to `agent_ros.log`; the terminal shows errors only |
| `test/object_scan_geometry_test.py` | Unit tests for the scan geometry |

## 🧪 Tests

```bash
PYTHONPATH=agent_orchestrator/src python3 -m unittest discover -s agent_orchestrator/test -p "*_test.py"
```

## ⚠️ Known outdated

`src/reset_rail.sh` publishes to `/direct_joint_command` and `/gripper/command`. The bridge now rejects these topics, so the script no longer moves the robot.
