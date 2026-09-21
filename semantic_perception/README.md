# 👁️ Semantic perception

A ROS 2 package that turns the wrist RGB-D camera into a persistent map of objects: detection (YOLO), appearance embeddings (MobileCLIP), 3-D localization with uncertainty, tracking, and storage in `semantic_objects.json`. It also caches views for Gaussian Splatting. Robot motion stays with the [orchestrator](../agent_orchestrator/README.md).

For how it fits into the whole system, see the [root README](../README.md).

## 🚀 Run

From the repository root, with the simulator running:

```bash
./run_semantic_perception.sh
```

| Flag | Effect |
|---|---|
| `--rviz` / `--rviz-confirmed-only` | Also starts the RViz marker bridge (confirmed-only hides candidates and stale objects) |
| `--publish-debug` | Publishes the annotated YOLO image on `/semantic/debug/yolo` |
| `--scene-export` | Saves scene keyframes for scene-level Gaussian Splatting |
| `-p name:=value` | Passed to the node as a ROS parameter |

Tuned command lines for the simulator and a real camera are in [`commands.txt`](../commands.txt).

**Environment variables** read by the script:

| Variable | Default |
|---|---|
| `ARENA_DETECTOR_MODEL` | `agent_orchestrator/models/yolo26x.pt` |
| `ARENA_MOBILECLIP_CHECKPOINT` | `agent_orchestrator/models/mobileclip_s0.pt` |
| `ARENA_OBJECT_REGISTRY` | `semantic_objects.json` |
| `ARENA_USE_SIM_TIME` | `true` (use `false` with a real wall-clock camera) |
| `ARENA_CONFIRMATION_HITS` | `8` |
| `ARENA_PROCESSING_RATE_HZ` | `12.0` |
| `ARENA_PUBLISH_ANNOTATED_DEBUG` | `false` |
| `ARENA_OBJECT_POINT_VOXEL_SIZE_M` | `0.02` |
| `ARENA_DYNAMIC_COORDINATES`, `ARENA_WRITE_LEGACY_COORDINATES` | legacy `semantic_distances_dynamic.json` export (off) |

You can also build it as a normal ROS package and use the launch file (`model_path` is required):

```bash
colcon build --base-paths semantic_perception --packages-select semantic_perception
source install/setup.bash
ros2 launch semantic_perception semantic_perception.launch.py model_path:=/path/to/yolo.pt
```

## 🛠️ Setup

- Build the `interface` package of the `home_robotics` workspace first (`colcon build --packages-select interface`). It provides `DetectedObject(Array)`, `FindObject`, `SearchSemanticObjects` and `AssociationDiagnosticArray`.
- Install `ultralytics` in the Python environment that starts the node.
- Install Apple's MobileCLIP and download the S0 checkpoint. The node never downloads weights and stops with a clear error if either is missing.

```bash
git clone https://github.com/apple/ml-mobileclip.git
./agent_orchestrator/agent_env/bin/pip install -e ./ml-mobileclip
```

## 📡 ROS API

| Name | Type | Purpose |
|---|---|---|
| `/semantic/objects` | `DetectedObjectArray` (latched) | The full registry. Late subscribers get the last map immediately |
| `/semantic/detections` | `DetectedObjectArray` | Objects localized from the current frame |
| `/semantic/target_found` | `DetectedObject` | Published when a `find_object` goal succeeds |
| `/semantic/find_object` | action `FindObject` | Succeeds on a **fresh** confirmed sighting of a class made after the goal started (class probability ≥ 0.70 and enough new hits) |
| `/semantic/search_objects` | service `SearchSemanticObjects` | Ranks confirmed objects against a text query with MobileCLIP (top 5 above cosine similarity 0.25). This is map evidence, not visual proof |
| `/semantic/association_diagnostics` | `AssociationDiagnosticArray` | Per-frame timing, depth statistics, covariance and match decision (`matched`, `new_track`, `depth_rejected`) |
| `/semantic/debug/yolo` | `Image` | Annotated detections, with `--publish-debug` |
| `/semantic/rviz/markers` | `MarkerArray` | From the RViz bridge. Fixed Frame: `global_origin` |

Inputs: the wrist color and depth images (`topics.color`, `topics.aligned_depth`) and `/tf`.

## 🔬 How it works

1. **Sync.** RGB and depth are paired within 30 ms. A packet waits until TF can be interpolated at the image time. The latest TF is not used by default (`tf.fallback_to_latest: false`), because it would smear positions while the camera moves.
2. **Detect and embed.** YOLO finds boxes. Boxes touching the image border or covering more than 55 % of the image are dropped. MobileCLIP-S0 embeds all crops of a frame in one batch.
3. **Localize.** A robust depth sample from the inner part of the box is back-projected with the fixed calibration into `global_origin`, with a 3×3 covariance that grows beyond 1 m.
4. **Match.** A detection joins a track only if it passes the 3-D Mahalanobis gate (99 %, `d² ≤ 11.345`) and the appearance gate (cosine distance ≤ 0.35). Matches are solved with the Hungarian algorithm.
5. **Update.** Position: Kalman update. Class: a soft distribution over at most four labels plus `other`. A track is *confirmed* after enough hits in the confirmation window (class probability ≥ 0.70) and goes *stale* after 300 s unseen.
6. **Save.** The registry is written to `semantic_objects.json` (schema v3) once per second. Older schemas and legacy coordinate files are migrated on load.

## ⚙️ Configuration

Defaults are in `config/semantic_perception.yaml`, tuned for the simulator. Use `config/semantic_perception_real.yaml` (more conservative noise values) for a real camera:

- set all six `camera.*` values (`fx`, `fy`, `cx`, `cy`, `image_width`, `image_height`) from your camera. The node has no `CameraInfo` input and rejects frames of a different size
- set `camera.tf_frame_override` to your camera's TF frame, or leave it empty to use the image header frame. The simulator labels images `sim_camera`, but its TF calls the frame `wrist_camera`
- run with `ARENA_USE_SIM_TIME=false` if the camera uses wall time

Simulator calibration: 1280×720, fx 907.00, fy 905.69, cx 567.74, cy 488.32.

Useful parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `detector.confidence_threshold` | 0.50 | Minimum YOLO confidence |
| `detector.max_bbox_area_fraction` | 0.55 | Drop boxes larger than this share of the image |
| `filter.confirmation_hits` / `filter.confirmation_window_sec` | 3 / 3.0 | Hits needed to confirm a track (the run script sets 8 hits) |
| `class.confirmation_probability` | 0.70 | Class probability needed to confirm |
| `depth.maximum_m` | 5.0 | Ignore depth beyond this range |
| `processing_rate_hz` | 12.0 | Detection rate |
| `diagnostics.publish` | true | Publish `/semantic/association_diagnostics` |
| `debug.timing` | false | Write per-stage timings (`debug.timing_log_path`) |

## 🌫️ Gaussian Splatting export

The node writes training data to the RAM disk `/dev/shm/3dgs_cache/`:
- **per object:** RGB and depth crops, camera pose, intrinsics and a sparse point cloud. A pose filter (`export.voxel_size_m`, `export.voxel_size_deg`) skips near-duplicate views
- **scene** (with `--scene-export`): keyframes every 10 cm or 8° of camera motion, at most 600, with background points

The [`gaussian_splatting_ros`](../gaussian_splatting_ros) node trains from this cache.

## 🐞 Debugging

```bash
# Show the YOLO overlay (start with --publish-debug)
python3 scripts/view_wrist_camera.py --topic /semantic/debug/yolo --scale 0.6

# Raw YOLO output without depth or tracking (do not run together with --publish-debug)
PYTHONPATH=semantic_perception python3 -m semantic_perception.yolo_debug --ros-args -p model_path:=/path/to/yolo.pt

# Watch matching decisions
ros2 topic echo /semantic/association_diagnostics
```

`yolo_debug` defaults to a lower confidence of 0.25 so marginal boxes stay visible.
