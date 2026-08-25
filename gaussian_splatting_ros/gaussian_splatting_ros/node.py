"""Gaussian Splatting ROS 2 Node.

Monitors the Ramdisk cache written by semantic_perception, compiles
training data into a unified transforms.json, launches the 3DGS
training subprocess, persists the resulting model, and cleans up.
"""

import json
import glob
import itertools
import os
import shutil
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from interface.srv import OptimizeObject
from .dataset import (
    frame_metadata_paths,
    load_json_if_available,
    ros_optical_to_nerfstudio_camera_to_world,
    spatial_voxel_filter,
    write_ascii_ply,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_DIR = "/dev/shm/3dgs_cache"
_DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Arena-RealSim-Data/3dgs_models")
_MIN_FRAMES_FOR_TRAINING = 30


class GaussianSplattingNode(Node):
    """Async 3DGS training node that reads from a Ramdisk cache."""

    def __init__(self):
        super().__init__("gaussian_splatting_node")

        # -- Parameters -------------------------------------------------------
        self.declare_parameter("cache_dir", _DEFAULT_CACHE_DIR)
        self.declare_parameter("output_dir", _DEFAULT_OUTPUT_DIR)
        self.declare_parameter("min_frames", _MIN_FRAMES_FOR_TRAINING)
        self.declare_parameter("training_iterations", 20000)
        self.declare_parameter("training_command", "ns-train")  # placeholder
        self.declare_parameter("scene_background_voxel_size_m", 0.03)
        self.declare_parameter("object_point_voxel_size_m", 0.01)
        self.declare_parameter("depth_supervision.enabled", True)
        self.declare_parameter("depth_supervision.loss_weight", 0.05)
        self.declare_parameter("depth_supervision.warmup_steps", 500)
        self.declare_parameter("depth_supervision.ramp_steps", 1500)
        self.declare_parameter("depth_supervision.huber_delta_m", 0.02)
        self.declare_parameter("depth_supervision.minimum_depth_m", 0.10)
        self.declare_parameter("depth_supervision.maximum_depth_m", 3.0)
        self.declare_parameter("depth_supervision.edge_threshold_m", 0.06)
        self.declare_parameter("depth_supervision.minimum_opacity", 0.05)

        self.cache_dir: str = self.get_parameter("cache_dir").value
        self.output_dir: str = self.get_parameter("output_dir").value
        self.min_frames: int = self.get_parameter("min_frames").value
        if float(self.get_parameter("scene_background_voxel_size_m").value) <= 0.0:
            raise ValueError("scene_background_voxel_size_m must be positive")
        if float(self.get_parameter("object_point_voxel_size_m").value) <= 0.0:
            raise ValueError("object_point_voxel_size_m must be positive")
        for name in (
            "depth_supervision.loss_weight",
            "depth_supervision.huber_delta_m",
            "depth_supervision.minimum_depth_m",
            "depth_supervision.maximum_depth_m",
            "depth_supervision.edge_threshold_m",
            "depth_supervision.minimum_opacity",
        ):
            if float(self.get_parameter(name).value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.get_parameter("depth_supervision.maximum_depth_m").value) <= float(
            self.get_parameter("depth_supervision.minimum_depth_m").value
        ):
            raise ValueError("depth supervision maximum depth must exceed minimum depth")
        for name in ("depth_supervision.warmup_steps", "depth_supervision.ramp_steps"):
            if int(self.get_parameter(name).value) < 0:
                raise ValueError(f"{name} must be non-negative")

        # -- State -------------------------------------------------------------
        self._training_lock = threading.Lock()
        self._active_trainings: dict[str, threading.Thread] = {}

        # -- ROS interfaces ----------------------------------------------------
        self.srv = self.create_service(
            OptimizeObject, "~/optimize_object", self.optimize_callback
        )
        self.timer = self.create_timer(2.0, self.monitor_cache_callback)

        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(
            f"Gaussian Splatting Node initialized. "
            f"Cache: {self.cache_dir} | Output: {self.output_dir}"
        )

    # -----------------------------------------------------------------------
    # Cache Monitor  (runs on a 2 Hz timer)
    # -----------------------------------------------------------------------
    def monitor_cache_callback(self) -> None:
        """Scan the Ramdisk cache and log readiness per object."""
        if not os.path.isdir(self.cache_dir):
            return

        for object_id in sorted(os.listdir(self.cache_dir)):
            obj_dir = os.path.join(self.cache_dir, object_id)
            if not os.path.isdir(obj_dir):
                continue

            num_frames = len(frame_metadata_paths(obj_dir))
            if num_frames > 0 and num_frames % 25 == 0:
                ready = num_frames >= self.min_frames
                status = "READY" if ready else "accumulating"
                self.get_logger().info(
                    f"[Monitor] {object_id}: {num_frames} views ({status})"
                )

    # -----------------------------------------------------------------------
    # Service  ~/optimize_object
    # -----------------------------------------------------------------------
    def optimize_callback(self, request, response):
        """Trigger training for a specific object, or auto-find one if empty."""
        if not os.path.isdir(self.cache_dir):
            response.success = False
            response.message = "Cache directory does not exist."
            return response

        target_id = request.object_id.strip() if request.object_id else None

        if target_id is None:
            # Auto-find the first object with enough frames
            for object_id in sorted(os.listdir(self.cache_dir)):
                obj_dir = os.path.join(self.cache_dir, object_id)
                if not os.path.isdir(obj_dir):
                    continue
                num_frames = len(frame_metadata_paths(obj_dir))
                if num_frames >= self.min_frames:
                    target_id = object_id
                    break

            if target_id is None:
                response.success = False
                response.message = (
                    f"No object has accumulated >= {self.min_frames} frames yet."
                )
                return response
        else:
            # Validate user-provided target_id
            obj_dir = os.path.join(self.cache_dir, target_id)
            if not os.path.isdir(obj_dir):
                response.success = False
                response.message = f"Cache directory {obj_dir} does not exist."
                return response
            
            num_frames = len(frame_metadata_paths(obj_dir))
            if num_frames < self.min_frames:
                self.get_logger().warning(f"Object {target_id} only has {num_frames}/{self.min_frames} frames. Triggering anyway!")

        # Prevent double-training
        with self._training_lock:
            if target_id in self._active_trainings:
                response.success = False
                response.message = f"{target_id} is already being trained."
                return response

            thread = threading.Thread(
                target=self._run_3dgs_training,
                args=(target_id,),
                daemon=True,
            )
            self._active_trainings[target_id] = thread
            thread.start()

        response.success = True
        response.message = f"Started background training for {target_id}."
        return response

    # -----------------------------------------------------------------------
    # Step 1 – Compile per-frame JSONs into a single transforms.json
    # -----------------------------------------------------------------------
    @staticmethod
    def _point_groups(metadata_paths: list[str]):
        for metadata_path in metadata_paths:
            payload = load_json_if_available(metadata_path)
            if payload is None:
                continue
            points = payload.get("point_cloud", [])
            if isinstance(points, list):
                yield points

    def _scene_object_point_groups(self):
        """Yield dense full-crop clouds from every live non-scene cache."""
        for object_id in sorted(os.listdir(self.cache_dir)):
            if object_id == "scene":
                continue
            object_dir = os.path.join(self.cache_dir, object_id)
            if not os.path.isdir(object_dir):
                continue
            yield from self._object_point_groups(object_dir)

    def _object_point_groups(self, object_dir: str):
        """Return one globally de-duplicated dense cloud for an object cache."""
        points = spatial_voxel_filter(
            (
                point
                for group in self._point_groups(frame_metadata_paths(object_dir))
                for point in group
            ),
            float(self.get_parameter("object_point_voxel_size_m").value),
        )
        if points:
            yield points

    def _compile_transforms(self, obj_dir: str, target_id: str) -> str:
        """Read individual <timestamp>.json files and merge them into
        a single ``transforms.json`` in Nerfstudio-compatible format.

        Returns the path to the written ``transforms.json``.
        """
        metadata_paths = frame_metadata_paths(obj_dir)
        frames = []
        for jf in metadata_paths:
            data = load_json_if_available(jf)
            if data is None:
                self.get_logger().warning(f"[Compile] Skipping unavailable metadata {jf}")
                continue
            timestamp = os.path.splitext(os.path.basename(jf))[0]
            png_path = os.path.join(obj_dir, f"{timestamp}.png")
            if not os.path.isfile(png_path):
                continue
            try:
                intrinsics = data["intrinsics"]
                frame = {
                    "file_path": png_path,
                    # Cache poses are ROS optical-frame c2w matrices.  The
                    # Nerfstudio pinhole camera emits OpenGL-frame rays, so
                    # convert the camera basis while retaining the identical
                    # global-origin translation and world-space seed cloud.
                    "transform_matrix": ros_optical_to_nerfstudio_camera_to_world(
                        data["camera_to_world"]
                    ),
                    "fl_x": intrinsics["fx"],
                    "fl_y": intrinsics["fy"],
                    "cx": intrinsics["cx"],
                    "cy": intrinsics["cy"],
                    "w": intrinsics["width"],
                    "h": intrinsics["height"],
                }
            except (KeyError, TypeError):
                self.get_logger().warning(f"[Compile] Skipping malformed metadata {jf}")
                continue

            # Include depth map if available
            depth_path = os.path.join(obj_dir, f"{timestamp}_depth.png")
            if os.path.isfile(depth_path):
                frame["depth_file_path"] = depth_path

            frames.append(frame)

        if not frames:
            raise RuntimeError(f"No complete training frames found in {obj_dir}")

        transforms = {
            "camera_model": "PINHOLE",
            "frames": frames,
        }

        ply_path = os.path.join(obj_dir, "points3D.ply")
        if target_id == "scene":
            background_points = spatial_voxel_filter(
                (
                    point
                    for group in self._point_groups(metadata_paths)
                    for point in group
                ),
                float(self.get_parameter("scene_background_voxel_size_m").value),
            )
            point_groups = iter((background_points,))
            point_groups = itertools.chain(
                point_groups, self._scene_object_point_groups()
            )
        else:
            point_groups = self._object_point_groups(obj_dir)
        point_count = write_ascii_ply(ply_path, point_groups)
        if point_count > 0:
            transforms["ply_file_path"] = "points3D.ply"

        out_path = os.path.join(obj_dir, "transforms.json")
        temporary_path = f"{out_path}.tmp"
        with open(temporary_path, "w") as fh:
            json.dump(transforms, fh, indent=2)
        os.replace(temporary_path, out_path)

        self.get_logger().info(
            f"[Compile] Wrote transforms.json with {len(frames)} frames "
            f"and {point_count} initialization points."
        )
        return out_path

    # -----------------------------------------------------------------------
    # Step 2 – Launch Training Subprocess
    # -----------------------------------------------------------------------
    def _launch_training(self, obj_dir: str, object_id: str) -> int:
        """Launch the 3DGS training engine as a subprocess.

        Returns the process exit code (0 = success).
        """
        iterations = self.get_parameter("training_iterations").value
        train_cmd = self.get_parameter("training_command").value

        depth_enabled = bool(self.get_parameter("depth_supervision.enabled").value)
        method = "depth-splatfacto" if depth_enabled else "splatfacto"
        cmd = [
            train_cmd,
            method,
            "--data", obj_dir,
            "--max-num-iterations", str(iterations),
            "--output-dir", os.path.join(self.output_dir, object_id),
            "--vis", "viewer+tensorboard",
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.datamanager.cache-images", "cpu",
            "--pipeline.datamanager.max-thread-workers", "4",
        ]
        environment = os.environ.copy()
        if depth_enabled:
            plugin = (
                "depth-splatfacto="
                "gaussian_splatting_ros.depth_splatfacto:depth_splatfacto_method"
            )
            existing = environment.get("NERFSTUDIO_METHOD_CONFIGS", "").strip()
            environment["NERFSTUDIO_METHOD_CONFIGS"] = (
                f"{existing},{plugin}" if existing else plugin
            )
            option_map = {
                "depth-loss-weight": "depth_supervision.loss_weight",
                "depth-warmup-steps": "depth_supervision.warmup_steps",
                "depth-ramp-steps": "depth_supervision.ramp_steps",
                "depth-huber-delta-m": "depth_supervision.huber_delta_m",
                "depth-minimum-m": "depth_supervision.minimum_depth_m",
                "depth-maximum-m": "depth_supervision.maximum_depth_m",
                "depth-edge-threshold-m": "depth_supervision.edge_threshold_m",
                "depth-minimum-opacity": "depth_supervision.minimum_opacity",
            }
            for option, parameter in option_map.items():
                cmd.extend([
                    f"--pipeline.model.{option}",
                    str(self.get_parameter(parameter).value),
                ])

        self.get_logger().info(f"[Train] Launching: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            # Stream stdout into the ROS logger
            for line in iter(proc.stdout.readline, ""):
                stripped = line.rstrip()
                if stripped:
                    self.get_logger().info(f"[Train] {stripped}")
            proc.wait()
            if proc.returncode != 0:
                return proc.returncode
                
            # After successful training, we must explicitly export the .ply model
            engine_out = os.path.join(self.output_dir, object_id)
            configs = glob.glob(os.path.join(engine_out, "**", "config.yml"), recursive=True)
            if not configs:
                self.get_logger().error("[Export] Training completed but produced no config.yml.")
                return -2
            best_config = max(configs, key=os.path.getmtime)
            export_cmd = [
                train_cmd.replace("ns-train", "ns-export"),
                "gaussian-splat",
                "--load-config", best_config,
                "--output-dir", engine_out
            ]
            self.get_logger().info(f"[Export] Launching PLY export: {' '.join(export_cmd)}")
            # ``ns-export`` loads the saved method name from config.yml in
            # a fresh process, so it needs the same local method registry
            # as the training subprocess.
            export_result = subprocess.run(export_cmd, check=False, env=environment)
            if export_result.returncode != 0:
                self.get_logger().error(
                    "[Export] PLY export failed (exit code %s); preserving cache for retry."
                    % export_result.returncode
                )
                return export_result.returncode
            if not glob.glob(os.path.join(engine_out, "**", "*.ply"), recursive=True):
                self.get_logger().error(
                    "[Export] Export exited successfully but produced no PLY; preserving cache for retry."
                )
                return -3
            return 0
        except FileNotFoundError:
            self.get_logger().error(
                f"[Train] Command '{train_cmd}' not found. "
                "Is the 3DGS engine installed and on PATH?"
            )
            return -1

    # -----------------------------------------------------------------------
    # Step 3 – Persist Model
    # -----------------------------------------------------------------------
    def _persist_model(self, object_id: str) -> bool:
        """Copy the trained .ply model out of the engine's output dir
        into a permanent location.

        Returns True if a model was found and copied.
        """
        engine_out = os.path.join(self.output_dir, object_id)
        # Search recursively for .ply files produced by the engine
        ply_files = glob.glob(os.path.join(engine_out, "**", "*.ply"), recursive=True)
        if not ply_files:
            self.get_logger().warning(
                f"[Persist] No .ply model found under {engine_out}."
            )
            return False

        # Take the most recently modified .ply
        best = max(ply_files, key=os.path.getmtime)
        dest = os.path.join(self.output_dir, f"{object_id}.ply")
        shutil.copy2(best, dest)
        self.get_logger().info(f"[Persist] Model saved to {dest}")
        return True

    # -----------------------------------------------------------------------
    # Step 4 – Cleanup
    # -----------------------------------------------------------------------
    def _cleanup_cache(self, object_id: str) -> None:
        """Reclaim cache after training; scene export consumes every cache source."""
        cleanup_path = (
            self.cache_dir if object_id == "scene"
            else os.path.join(self.cache_dir, object_id)
        )
        if os.path.isdir(cleanup_path):
            shutil.rmtree(cleanup_path)
            self.get_logger().info(
                "[Cleanup] Removed %s from Ramdisk after %s training."
                % (cleanup_path, object_id)
            )

    # -----------------------------------------------------------------------
    # Orchestrator  (runs in background thread)
    # -----------------------------------------------------------------------
    def _run_3dgs_training(self, object_id: str) -> None:
        """Full pipeline: compile ➜ train ➜ persist ➜ cleanup."""
        obj_dir = os.path.join(self.cache_dir, object_id)
        self.get_logger().info(
            f"[Thread] Starting 3DGS pipeline for {object_id}..."
        )

        try:
            # Step 1 – Compile transforms.json
            self._compile_transforms(obj_dir, object_id)

            # Step 2 – Launch training subprocess
            exit_code = self._launch_training(obj_dir, object_id)
            if exit_code != 0:
                self.get_logger().error(
                    f"[Thread] Training failed for {object_id} "
                    f"(exit code {exit_code}). Skipping persist & cleanup."
                )
                return

            # Step 3 – Persist the .ply model
            self._persist_model(object_id)

            # Step 4 – Cleanup Ramdisk
            self._cleanup_cache(object_id)

            self.get_logger().info(
                f"[Thread] 3DGS pipeline completed for {object_id}!"
            )
        except Exception as exc:
            self.get_logger().error(
                f"[Thread] 3DGS pipeline crashed for {object_id}: {exc}"
            )
        finally:
            with self._training_lock:
                self._active_trainings.pop(object_id, None)


def main(args=None):
    rclpy.init(args=args)
    node = GaussianSplattingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Gaussian Splatting Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
