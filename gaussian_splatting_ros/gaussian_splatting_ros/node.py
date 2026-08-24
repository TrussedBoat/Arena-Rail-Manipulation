"""Gaussian Splatting ROS 2 Node.

Monitors the Ramdisk cache written by semantic_perception, compiles
training data into a unified transforms.json, launches the 3DGS
training subprocess, persists the resulting model, and cleans up.
"""

import json
import glob
import os
import shutil
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from interface.srv import OptimizeObject


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
        self.declare_parameter("training_iterations", 30000)
        self.declare_parameter("training_command", "ns-train")  # placeholder

        self.cache_dir: str = self.get_parameter("cache_dir").value
        self.output_dir: str = self.get_parameter("output_dir").value
        self.min_frames: int = self.get_parameter("min_frames").value

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

            num_frames = len(glob.glob(os.path.join(obj_dir, "*.png")))
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
                num_frames = len(glob.glob(os.path.join(obj_dir, "*.png")))
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
            
            num_frames = len(glob.glob(os.path.join(obj_dir, "*.png")))
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
    def _compile_transforms(self, obj_dir: str) -> str:
        """Read individual <timestamp>.json files and merge them into
        a single ``transforms.json`` in Nerfstudio-compatible format.

        Returns the path to the written ``transforms.json``.
        """
        json_files = sorted(glob.glob(os.path.join(obj_dir, "*.json")))
        # Exclude any previously written transforms.json
        json_files = [f for f in json_files if not f.endswith("transforms.json")]

        frames = []
        all_points = []
        
        for jf in json_files:
            with open(jf, "r") as fh:
                data = json.load(fh)
            timestamp = os.path.splitext(os.path.basename(jf))[0]
            png_path = os.path.join(obj_dir, f"{timestamp}.png")
            if not os.path.isfile(png_path):
                continue

            frame = {
                "file_path": png_path,
                "transform_matrix": data["camera_to_world"],
                "fl_x": data["intrinsics"]["fx"],
                "fl_y": data["intrinsics"]["fy"],
                "cx": data["intrinsics"]["cx"],
                "cy": data["intrinsics"]["cy"],
                "w": data["intrinsics"]["width"],
                "h": data["intrinsics"]["height"],
            }

            # Include depth map if available
            depth_path = os.path.join(obj_dir, f"{timestamp}_depth.png")
            if os.path.isfile(depth_path):
                frame["depth_file_path"] = depth_path

            # Aggregate point cloud samples
            if "point_cloud" in data and data["point_cloud"]:
                all_points.extend(data["point_cloud"])

            frames.append(frame)

        transforms = {
            "camera_model": "PINHOLE",
            "frames": frames,
        }

        # Write PLY file if we have point clouds
        if all_points:
            ply_path = os.path.join(obj_dir, "points3D.ply")
            with open(ply_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(all_points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for pt in all_points:
                    f.write(f"{pt['x']} {pt['y']} {pt['z']} {pt['r']} {pt['g']} {pt['b']}\n")
            
            transforms["ply_file_path"] = "points3D.ply"

        out_path = os.path.join(obj_dir, "transforms.json")
        with open(out_path, "w") as fh:
            json.dump(transforms, fh, indent=2)

        self.get_logger().info(
            f"[Compile] Wrote transforms.json with {len(frames)} frames."
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

        cmd = [
            train_cmd,
            "splatfacto",
            "--data", obj_dir,
            "--max-num-iterations", str(iterations),
            "--output-dir", os.path.join(self.output_dir, object_id),
            "--vis", "viewer+tensorboard",
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.datamanager.cache-images", "cpu",
            "--pipeline.datamanager.max-thread-workers", "4",
        ]

        self.get_logger().info(f"[Train] Launching: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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
            if configs:
                best_config = max(configs, key=os.path.getmtime)
                export_cmd = [
                    train_cmd.replace("ns-train", "ns-export"),
                    "gaussian-splat",
                    "--load-config", best_config,
                    "--output-dir", engine_out
                ]
                self.get_logger().info(f"[Export] Launching PLY export: {' '.join(export_cmd)}")
                subprocess.run(export_cmd, check=False)
                
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
        """Remove the Ramdisk cache for this object to reclaim RAM."""
        obj_dir = os.path.join(self.cache_dir, object_id)
        if os.path.isdir(obj_dir):
            shutil.rmtree(obj_dir)
            self.get_logger().info(
                f"[Cleanup] Removed cache for {object_id} from Ramdisk."
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
            self._compile_transforms(obj_dir)

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
