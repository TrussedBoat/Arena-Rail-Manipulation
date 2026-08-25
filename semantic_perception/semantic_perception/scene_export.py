"""Pure geometry helpers for scene-level 3DGS capture."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def camera_pose_voxel(
    camera_to_world: np.ndarray,
    translation_size_m: float,
    rotation_size_deg: float,
) -> tuple[int, int, int, int, int, int]:
    """Quantize a camera pose into a 6D translation/orientation keyframe bin."""
    if camera_to_world.shape != (4, 4) or not np.all(np.isfinite(camera_to_world)):
        raise ValueError("camera_to_world must be a finite 4x4 matrix")
    if translation_size_m <= 0.0 or rotation_size_deg <= 0.0:
        raise ValueError("scene keyframe voxel sizes must be positive")
    rotation = camera_to_world[:3, :3]
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    pitch = math.atan2(
        -rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2])
    )
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    rotation_size_rad = math.radians(rotation_size_deg)
    tx, ty, tz = camera_to_world[:3, 3]
    return (
        math.floor(float(tx) / translation_size_m),
        math.floor(float(ty) / translation_size_m),
        math.floor(float(tz) / translation_size_m),
        math.floor(roll / rotation_size_rad),
        math.floor(pitch / rotation_size_rad),
        math.floor(yaw / rotation_size_rad),
    )


def sample_scene_background_points(
    *,
    color: np.ndarray,
    depth: np.ndarray,
    detection_boxes: Iterable[tuple[float, float, float, float]],
    camera_matrix: list[float],
    camera_to_world: np.ndarray,
    minimum_depth_m: float,
    maximum_depth_m: float,
    maximum_range_m: float,
    maximum_points: int,
    box_padding_fraction: float,
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    """Sample valid background pixels outside padded YOLO rectangles."""
    if color.ndim != 3 or color.shape[:2] != depth.shape:
        raise ValueError("scene RGB and depth images must be aligned")
    if len(camera_matrix) != 9:
        raise ValueError("camera matrix must contain nine values")
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")
    if (
        maximum_points <= 0
        or maximum_range_m <= 0.0
        or not 0.0 <= box_padding_fraction <= 1.0
    ):
        raise ValueError("invalid scene sampling configuration")

    height, width = depth.shape
    mask = (
        np.isfinite(depth)
        & (depth >= minimum_depth_m)
        & (depth <= maximum_depth_m)
    )
    for x1, y1, x2, y2 in detection_boxes:
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        pad_x = box_width * box_padding_fraction
        pad_y = box_height * box_padding_fraction
        left = max(0, min(width, math.floor(x1 - pad_x)))
        right = max(0, min(width, math.ceil(x2 + pad_x)))
        top = max(0, min(height, math.floor(y1 - pad_y)))
        bottom = max(0, min(height, math.ceil(y2 + pad_y)))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = False

    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return []
    if rows.size > maximum_points:
        selected = rng.choice(rows.size, size=maximum_points, replace=False)
        rows = rows[selected]
        columns = columns[selected]

    z = depth[rows, columns].astype(np.float64, copy=False)
    fx, fy = float(camera_matrix[0]), float(camera_matrix[4])
    cx, cy = float(camera_matrix[2]), float(camera_matrix[5])
    camera_points = np.column_stack((
        (columns.astype(np.float64) - cx) * z / fx,
        (rows.astype(np.float64) - cy) * z / fy,
        z,
    ))
    within_range = np.linalg.norm(camera_points, axis=1) <= maximum_range_m
    camera_points = camera_points[within_range]
    rows = rows[within_range]
    columns = columns[within_range]
    if camera_points.size == 0:
        return []
    world_points = (
        camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    )
    bgr = color[rows, columns]
    return [
        {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "r": int(pixel[2]),
            "g": int(pixel[1]),
            "b": int(pixel[0]),
        }
        for point, pixel in zip(world_points, bgr, strict=True)
    ]


def voxel_downsample_points(
    points: Iterable[dict[str, float | int]], voxel_size_m: float
) -> list[dict[str, float | int]]:
    """Keep one colored point per finite world-space voxel."""
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    voxels: dict[tuple[int, int, int], dict[str, float | int]] = {}
    for point in points:
        x, y, z = (float(point[axis]) for axis in ("x", "y", "z"))
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        voxels.setdefault(
            (
                math.floor(x / voxel_size_m),
                math.floor(y / voxel_size_m),
                math.floor(z / voxel_size_m),
            ),
            point,
        )
    return list(voxels.values())
