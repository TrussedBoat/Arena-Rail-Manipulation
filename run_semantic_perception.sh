#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

VENV_PATH="$SCRIPT_DIR/agent_orchestrator/agent_env/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: agent environment not found: $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"
source /opt/ros/humble/setup.bash

HOME_ROBOTICS_SETUP="${ARENA_HOME_ROBOTICS_SETUP:-$HOME/workspace/home_robotics/homerobotics_ws/install/setup.bash}"
if [ ! -f "$HOME_ROBOTICS_SETUP" ]; then
    echo "Error: home_robotics setup not found: $HOME_ROBOTICS_SETUP"
    exit 1
fi
source "$HOME_ROBOTICS_SETUP"

MODEL_PATH="${ARENA_DETECTOR_MODEL:-$SCRIPT_DIR/agent_orchestrator/models/yolo26x.pt}"
MOBILECLIP_CHECKPOINT="${ARENA_MOBILECLIP_CHECKPOINT:-$SCRIPT_DIR/agent_orchestrator/models/mobileclip_s0.pt}"
REGISTRY_PATH="${ARENA_OBJECT_REGISTRY:-$SCRIPT_DIR/semantic_objects.json}"
LEGACY_PATH="${ARENA_DYNAMIC_COORDINATES:-$SCRIPT_DIR/semantic_distances_dynamic.json}"
WRITE_LEGACY="${ARENA_WRITE_LEGACY_COORDINATES:-false}"
# Arena publishes /clock, so simulation time is the safe default. Override
# with ARENA_USE_SIM_TIME=false when consuming a real wall-time camera setup.
USE_SIM_TIME="${ARENA_USE_SIM_TIME:-true}"
PUBLISH_ANNOTATED_DEBUG="${ARENA_PUBLISH_ANNOTATED_DEBUG:-false}"
CONFIRMATION_HITS="${ARENA_CONFIRMATION_HITS:-8}"
PROCESSING_RATE_HZ="${ARENA_PROCESSING_RATE_HZ:-12.0}"
OBJECT_POINT_VOXEL_SIZE_M="${ARENA_OBJECT_POINT_VOXEL_SIZE_M:-0.02}"
START_RVIZ_VISUALIZER=false
RVIZ_CONFIRMED_ONLY=false
EXPORT_SCENE=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --publish-debug)
            PUBLISH_ANNOTATED_DEBUG=true
            shift
            ;;
        --rviz)
            START_RVIZ_VISUALIZER=true
            shift
            ;;
        --rviz-confirmed-only)
            START_RVIZ_VISUALIZER=true
            RVIZ_CONFIRMED_ONLY=true
            shift
            ;;
        --scene-export)
            EXPORT_SCENE=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

export PYTHONPATH="$SCRIPT_DIR/semantic_perception${PYTHONPATH:+:$PYTHONPATH}"

if [ "$START_RVIZ_VISUALIZER" = true ]; then
    python3 -m semantic_perception.rviz_visualizer --ros-args \
      -p confirmed_only:="$RVIZ_CONFIRMED_ONLY" &
    RVIZ_VISUALIZER_PID=$!
    trap 'kill "$RVIZ_VISUALIZER_PID" 2>/dev/null || true' EXIT INT TERM
fi

python3 -m semantic_perception.node --ros-args \
  -p use_sim_time:="$USE_SIM_TIME" \
  -p detector.model_path:="$MODEL_PATH" \
  -p appearance.mobileclip_checkpoint:="$MOBILECLIP_CHECKPOINT" \
  -p registry.path:="$REGISTRY_PATH" \
  -p registry.legacy_coordinates_path:="$LEGACY_PATH" \
  -p registry.write_legacy_coordinates:="$WRITE_LEGACY" \
  -p debug.publish_annotated:="$PUBLISH_ANNOTATED_DEBUG" \
  -p filter.confirmation_hits:="$CONFIRMATION_HITS" \
  -p processing_rate_hz:="$PROCESSING_RATE_HZ" \
  -p export.voxel_size_m:=0.05 \
  -p export.voxel_size_deg:=5.0 \
  -p export.object_point_voxel_size_m:="$OBJECT_POINT_VOXEL_SIZE_M" \
  -p export.maximum_sample_range_m:=3.0 \
  -p export.scene.enabled:="$EXPORT_SCENE" \
  "$@"
