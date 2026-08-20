# Semantic perception

This ROS 2 package owns wrist RGB-D object detection, timestamped 3-D
localization, covariance-aware tracking, and semantic-object persistence.
Robot motion and search trajectories remain owned by the orchestrator.

## Build

Build the updated `interface` package first:

```bash
cd ../home_robotics/homerobotics_ws
colcon build --packages-select interface
source install/setup.bash
```

The Arena package can then run directly with the supplied script, or be built
as a normal ROS package:

```bash
cd Arena-RealSim
colcon build --base-paths semantic_perception --packages-select semantic_perception
source install/setup.bash
```

Ultralytics is an inference-time dependency and must be installed in the Python
environment used to start the node.

## Run

```bash
./run_semantic_perception.sh
```

## Visual YOLO debugging

Run this alongside the simulator to publish the raw detector output as an
annotated image, independent of RGB-D localization and registry confirmation:

```bash
./run_yolo_debug.sh
```

It publishes `sensor_msgs/Image` on `/semantic/debug/yolo`. To display it in
the included OpenCV viewer, use:

```bash
python3 scripts/view_wrist_camera.py --topic /semantic/debug/yolo
```

The debug node defaults to `0.25` so marginal boxes remain visible. Semantic
perception defaults to `0.50`; change either with `-p confidence_threshold:=…`.

Semantic perception rejects a detection when its bounding box covers more than
55% of the image (`detector.max_bbox_area_fraction`). This suppresses common
surface-sized false positives; increase the value for genuinely large nearby
objects, or set it to `1.0` to disable the filter. Class evidence decays by
`0.95` per observation, keeping roughly the last 20 observations influential.

To publish annotated frames from the semantic-perception node itself, start it
with `--publish-debug`:

```bash
./run_semantic_perception.sh --publish-debug
```

This publishes `/semantic/debug/yolo` using the exact YOLO inference result
that feeds localization. It is disabled by default. Do not run
`run_yolo_debug.sh` at the same time, because both nodes would publish to that
topic.

## RViz detections

Start the marker bridge alongside semantic perception:

```bash
./run_semantic_rviz_visualizer.sh
```

Or start it together with semantic perception using the launcher option:

```bash
./run_semantic_perception.sh --rviz
```

Use `--rviz-confirmed-only` instead to hide candidates, ambiguous tracks, and
stale objects from RViz.

In RViz, set **Fixed Frame** to `global_origin`, then add a **MarkerArray**
display with topic `/semantic/rviz/markers`. Green spheres are persistent,
confirmed objects from `/semantic/objects`; their text labels show class,
confidence, and state. The bridge receives the registry's transient-local
snapshot even when it starts after semantic perception. Markers persist in
RViz until the registry changes.

`ARENA_DETECTOR_MODEL`, `ARENA_OBJECT_REGISTRY`,
`ARENA_DYNAMIC_COORDINATES`, and `ARENA_WRITE_LEGACY_COORDINATES` override the
script defaults. The script uses wall time by default. `run_sim.sh` creates an
Isaac `/clock` publisher and configures ROS camera helpers to use simulation
timestamps. After restarting the simulator, use `ARENA_USE_SIM_TIME=true` and
`-p tf.fallback_to_latest:=false`. A built package may instead be started through
`semantic_perception.launch.py`.

The default `semantic_perception.yaml` uncertainty values are tuned for the
simulator. Pass `config_file:=.../semantic_perception_real.yaml` to select the
more conservative real-camera covariance floors.

The node uses fixed, rectified wrist-camera calibration from the `camera.*`
parameters; it does not require a `CameraInfo` topic. The simulator profile is
calibrated for `1280 × 720`, `fx=907.00`, `fy=905.69`, `cx=567.74`, and
`cy=488.32`. RGB and depth must be aligned and match that configured resolution.
The node rejects mismatched frames instead of producing incorrect coordinates.

For a real camera, set all six `camera.*` parameters from that camera's own
calibration and native aligned RGB-D resolution before starting the node.
The simulator image stream labels frames as `sim_camera`, but its TF tree calls
the camera `wrist_camera`; `camera.tf_frame_override` bridges that difference.
For real hardware set the override to its published TF camera frame, or empty
to use the image header's `frame_id` directly.

When image and TF timestamps come from different time domains, the simulator
profile falls back to the latest available TF transform (`tf.fallback_to_latest`)
after an exact lookup fails. Keep this enabled only while those clocks differ;
timestamped TF is preferred when both streams share a clock.

## ROS API

- `/semantic/detections`: localized observations from the current RGB-D packet.
- `/semantic/objects`: filtered registry with transient-local durability.
- `/semantic/target_found`: confirmed detections for simple event consumers.
- `/semantic/find_object`: targeted-search action. Success requires a fresh,
  confirmed observation after the goal starts.

Each object carries a normalized soft class distribution containing at most
four explicit labels plus `other`. `class_name` is the highest-probability
explicit label, or `unknown` when `other` is largest. Target search additionally
requires the representative class probability to be at least `0.70`.

Spatial association uses the full measurement and track covariance with the
3-D 99% chi-square Mahalanobis gate (`d² <= 11.345`). Eligible matches are
chosen globally with Hungarian assignment. By default, every valid YOLO crop in
a frame is batched through the image encoder of pretrained MobileCLIP-S0, then
L2-normalized. Incompatible appearance pairs are rejected above cosine distance
`0.35`, and the remaining assignment cost combines normalized `d²` and cosine
distance equally. The `/semantic/search_objects` service additionally reuses MobileCLIP-S0's text
encoder to rank compatible confirmed tracks for a natural-language description.
It returns at most five tracks above cosine similarity `0.25` by default; this
is retrieval evidence, not visual confirmation for manipulation.

Install Apple's official `mobileclip` package in the ROS Python environment
(`git clone https://github.com/apple/ml-mobileclip.git && ./agent_orchestrator/agent_env/bin/pip install -e ./ml-mobileclip`)
and download the MobileCLIP-S0 checkpoint before starting the node. Set
`ARENA_MOBILECLIP_CHECKPOINT=/absolute/path/to/mobileclip_s0.pt`, or pass
`-p appearance.mobileclip_checkpoint:=/absolute/path/to/mobileclip_s0.pt`.
The node never downloads model weights at runtime and reports a clear startup
error if the package or checkpoint is missing. Set `appearance.enabled:=false`
to return to spatial-only matching; `mobilenet_v3_small` and
`ultralytics_crop` remain available through `appearance.backend`.

Distance weighting treats observations closer than 1 m as full quality. Beyond
that distance, the world-frame covariance is inflated quadratically and class
evidence is down-weighted inversely with distance, with a 0.25 minimum evidence
weight. Configure `distance_weighting.reference_distance_m` and
`distance_weighting.minimum_class_evidence_weight` for a different camera range.

The node publishes `/semantic/association_diagnostics` for every processed
frame. It records TF mode and timestamp offset, depth/MAD sampling statistics,
box-derived pixel uncertainty, final covariance, and the selected association
decision/cost. `uncertainty.bbox_diagonal_fraction` (default `0.10`) adds
10% of the 2-D box diagonal in quadrature with base pixel uncertainty.
Inspect it with:

```bash
ros2 topic echo /semantic/association_diagnostics
```

`association_decision=matched` means an existing track was updated;
`new_track` means no eligible existing track survived the spatial and appearance
gates; `depth_rejected` includes the depth-sampling rejection reason.
Disable this diagnostic stream with `-p diagnostics.publish:=false`.

Detections that touch the configured camera-edge margin are rejected before
localization and appearance embedding. The default margin is 3% of each image
dimension (`detector.border_margin_fraction=0.03`). With `--publish-debug`, the
safe image region is magenta and rejected clipped boxes are red.

YOLO confidence is not treated as `P(other)`. For an accepted detection with
confidence `c`, its conditional class distribution is `class = 0.60 + 0.40c`
and `other = 1 - class`; its total evidence contribution is still scaled by
`c` and distance. Adjust the `0.60` floor with
`class.conditional_reliability_floor`.
Matched positions and covariances are fused with a static Kalman update.

`semantic_objects.json` schema version 3 stores the class distribution, 3×3
world-frame covariance, and the compatible appearance embedding when available.
Schema versions 1–2 and legacy coordinate files are migrated when loaded. The node
can also export the existing class-keyed `semantic_distances_dynamic.json`
format by setting `registry.write_legacy_coordinates:=true`. It defaults to
false until the old orchestrator writer is removed, preventing two processes
from racing to overwrite the same file during migration.
