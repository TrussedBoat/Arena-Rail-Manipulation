# 🤖 Robotic Task Orchestrator

A multimodal ROS 2 agent that captures environmental state via camera and orchestrates complex robotic subtasks using Vision-Language Models (VLMs).

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

- **Multimodal Intelligence**: Uses `Qwen3.6-35B-A3B-UD-IQ2_M` VLM running via `llama.cpp` server to analyze real-time environment state.
- **ROS 2 Integration**: Built-in image subscriber for direct communication with robotic sensors.
- **LangGraph Workflow**: State-of-the-art orchestration using directed graphs for robust execution.
- **Memory Support**: Short-term memory ability using `InMemorySaver` to track conversation context.
- **Dynamic Task Planning**: Decoupled design with dedicated agentic tools to query semantic positions, safely home/move joint states, and manipulate targets.

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

## 🎮 Running the Orchestrator (Neo-Classic Pipeline)

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

## 🎮 Running the Orchestrator (VLA)

To run the full system, open two separate terminals:

### Terminal 1: Start the Arena Simulation
```bash
cd Workspace/Arena-RealSim
./scripts/start_arena.sh rail openvla
```

### Terminal 2: Run the Orchestrator Agent
```bash
cd Workspace/Arena-RealSim
source agent_orchestrator/agent_orchestrator/bin/activate
source /opt/ros/humble/setup.bash
python3 agent_orchestrator/src/main.py --vla 'openvla'

## 📂 Project Structure

- **`src/main.py`**: Main entry point and interactive CLI runtime loop.
- **`src/agent.py`**: LangGraph setup, node configurations, and workflow compilation.
- **`src/tools.py`**: Agentic tool wrappers and execution modules.
- **`src/ros_interface.py`**: ROS 2 Node integration, subscription handlers, and hardware actuation interface.
