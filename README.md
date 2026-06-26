# 🏠 Home Robotics — Arena RealSim

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue)](https://docs.ros.org/en/humble/index.html)
[![Isaac Sim 5.0](https://img.shields.io/badge/Isaac%20Sim-5.0-orange)](https://developer.nvidia.com/isaac-sim)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)

A high-fidelity simulation environment for the **Home Robotics Arena**, built on NVIDIA Isaac Sim 5.0. This repository provides a modular, scriptable interface for controlling articulated assets, tables, and camera streams.

---

## 🌟 Features

### 🏢 Articulated Assets
- **Smart Furniture**: Interactive tables with elevation and depression control.
- **White Goods**: Operational fridge, oven, dishwasher, and washing machine doors.
- **Precision Rails**: Adjustable rail systems for moving robot bases across the arena.

### 🎥 Multimedia & Vision
- **TV Streaming**: Dynamically project any ROS 2 image topic onto the wall-mounted TV.
- **Multi-Camera Setup**: Integrated Franka wrist cameras, side cameras, and pick-and-place perspectives.

### 🛠️ Developer Tools
- **Modular Architecture**: Clean separation between ROS nodes, UI, and device configurations.
- **CLI Argparse**: Easy overrides for scene paths (`--scene`) and headless mode (`--headless`).
- **Command Clamping**: Automatic protection against out-of-bounds joint movements.

---

## 📂 Project Structure

```bash
scripts/
├── arena/               # Core logic package
│   ├── config.py       # Device & simulation parameters
│   ├── node.py         # ArenaController ROS 2 Node
│   ├── streamer.py     # TV Streaming logic
│   └── ui.py           # CLI Helper functions
├── arena.py            # Simulation entry point
├── arena_controller.py # Interactive CLI controller
└── start_arena.sh      # Unified startup script
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository into your workspace:
```bash
git clone https://github.com/romerhomerobotics/Arena-RealSim.git
cd Arena-RealSim
```

### 2. Launch the Simulation
Use the provided startup script to initialize the Isaac Sim environment:
```bash
./scripts/start_arena.sh [optional_isaac_python_path]
```
*Note: Default Isaac Python path is `~/isaac-sim/python.sh`.*

### 3. Run the Controller
In a separate terminal, launch the interactive command-line interface:
```bash
python3 scripts/arena_controller.py
```

---

## ⚙️ Configuration

- **Devices**: All device topics, limits, and units are defined in `scripts/arena/config.py`.
- **Scene**: Change the default USD scene via the `--scene` flag in `arena.py`.

---

© 2026 Home Robotics. All rights reserved.
