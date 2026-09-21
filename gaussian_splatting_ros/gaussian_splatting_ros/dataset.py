"""Dataset compilation helpers independent of ROS."""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
from collections.abc import Iterable


_NON_FRAME_JSON = {"transforms.json", "voxels.json", "scene_keyframes.json"}


def ros_optical_to_nerfstudio_camera_to_world(
    camera_to_world: list[list[float]],
) -> list[list[float]]:
    """Convert a ROS optical-frame c2w pose to Nerfstudio's OpenGL convention.

    ROS/OpenCV optical coordinates are ``(+x right, +y down, +z forward)``.
    Nerfstudio generates perspective rays in OpenGL coordinates
    ``(+x right, +y up, +z backward)``.  Cached RGB-D points are already in
    world coordinates and must *not* be changed; only the camera basis in the
    c2w matrix is converted by right-multiplying ``diag(1, -1, -1, 1)``.
    """
    if not isinstance(camera_to_world, list) or len(camera_to_world) != 4:
        raise ValueError("camera_to_world must be a 4x4 matrix")
    converted: list[list[float]] = []
    for row in camera_to_world:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("camera_to_world must be a 4x4 matrix")
        try:
            values = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise ValueError("camera_to_world must contain numeric values") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera_to_world must contain finite values")
        converted.append([values[0], -values[1], -values[2], values[3]])
    return converted


def frame_metadata_paths(directory: str) -> list[str]:
    """Snapshot complete frame metadata paths, ignoring sidecar JSON files."""
    paths: list[str] = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        name = os.path.basename(path)
        if name in _NON_FRAME_JSON:
            continue
        stem = os.path.splitext(name)[0]
        if os.path.isfile(os.path.join(directory, f"{stem}.png")):
            paths.append(path)
    return paths


def load_json_if_available(path: str) -> dict | None:
    """Read one live-cache JSON, returning None if it vanished or is incomplete."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def spatial_voxel_filter(
    points: Iterable[dict[str, float | int]], voxel_size_m: float
) -> list[dict[str, float | int]]:
    """Keep the first finite scene-background point in each world voxel."""
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    occupied: dict[tuple[int, int, int], dict[str, float | int]] = {}
    for point in points:
        try:
            x, y, z = (float(point[axis]) for axis in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        key = (
            math.floor(x / voxel_size_m),
            math.floor(y / voxel_size_m),
            math.floor(z / voxel_size_m),
        )
        occupied.setdefault(key, point)
    return list(occupied.values())


def write_ascii_ply(
    path: str, point_groups: Iterable[Iterable[dict[str, float | int]]]
) -> int:
    """Stream point groups into an atomic ASCII PLY without retaining all points."""
    body_path = f"{path}.body.tmp"
    output_path = f"{path}.tmp"
    count = 0
    try:
        with open(body_path, "w", encoding="utf-8") as body:
            for group in point_groups:
                for point in group:
                    try:
                        x, y, z = (float(point[axis]) for axis in ("x", "y", "z"))
                        red, green, blue = (
                            max(0, min(255, int(point[channel])))
                            for channel in ("r", "g", "b")
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not all(math.isfinite(value) for value in (x, y, z)):
                        continue
                    body.write(f"{x} {y} {z} {red} {green} {blue}\n")
                    count += 1
        with open(output_path, "w", encoding="utf-8") as output:
            output.write("ply\nformat ascii 1.0\n")
            output.write(f"element vertex {count}\n")
            output.write("property float x\nproperty float y\nproperty float z\n")
            output.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            output.write("end_header\n")
            with open(body_path, "r", encoding="utf-8") as body:
                shutil.copyfileobj(body, output)
        os.replace(output_path, path)
    finally:
        for temporary in (body_path, output_path):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return count
