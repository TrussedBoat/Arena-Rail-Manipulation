"""Geometry helpers for the RRT wrist-camera desk scan."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ScanPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class DeskTarget:
    """A desk-surface target expressed in panda_link0 coordinates."""

    global_x: float
    x: float
    y: float
    z: float


def rail_centre(rail_min: float, rail_max: float) -> float:
    if not all(math.isfinite(value) for value in (rail_min, rail_max)):
        raise ValueError("rail bounds must be finite")
    if rail_min >= rail_max:
        raise ValueError("rail_min must be less than rail_max")
    return (rail_min + rail_max) / 2.0


def generate_desk_arc(
    *,
    current_rail: float,
    rail_min: float,
    rail_max: float,
    side: int,
    desk_width: float,
    radius: float,
    height: float,
    roll: float,
    pitch: float,
    viewpoints: int,
) -> list[ScanPose]:
    """Return an inward arc whose end bearings point at the desk X corners.

    ``rail_min`` and ``rail_max`` are desk-corner X coordinates in the fixed,
    rail-aligned global frame. ``side`` is +1 for the robot's left row and -1
    for its right row.  The generated poses are expressed in ``panda_link0``,
    matching ``/rrt/pose_command``.

    ``panda_link0`` is rotated 180 degrees about Z from the global frame, so a
    global X corner must be converted before it is used as an RRT bearing:
    ``x_base = current_rail - x_global``.  (The lateral desk offset is already
    specified by ``side`` in the robot frame.)
    """
    values = (current_rail, rail_min, rail_max, desk_width, radius, height, roll, pitch)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scan geometry values must be finite")
    if rail_min >= rail_max:
        raise ValueError("rail_min must be less than rail_max")
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    if desk_width <= 0 or radius <= 0 or height <= 0:
        raise ValueError("desk width, radius, and height must be positive")
    if viewpoints < 2:
        raise ValueError("viewpoints must be at least 2")

    lateral = side * desk_width
    min_corner_x_base = current_rail - rail_min
    max_corner_x_base = current_rail - rail_max
    start_angle = math.atan2(lateral, min_corner_x_base)
    end_angle = math.atan2(lateral, max_corner_x_base)
    return [
        ScanPose(
            x=radius * math.cos(angle),
            y=radius * math.sin(angle),
            z=height,
            roll=roll,
            pitch=pitch,
            yaw=angle,
        )
        for angle in (
            start_angle + (end_angle - start_angle) * index / (viewpoints - 1)
            for index in range(viewpoints)
        )
    ]


def generate_desk_targets(
    *,
    current_rail: float,
    rail_min: float,
    rail_max: float,
    side: int,
    table_scan_y: float,
    surface_z: float,
    viewpoints: int,
) -> list[DeskTarget]:
    """Return evenly spaced global-X desk points in panda_link0 coordinates."""
    values = (current_rail, rail_min, rail_max, table_scan_y, surface_z)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("desk target values must be finite")
    if rail_min >= rail_max:
        raise ValueError("rail_min must be less than rail_max")
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    if table_scan_y <= 0:
        raise ValueError("table scan Y must be positive")
    if viewpoints < 2:
        raise ValueError("viewpoints must be at least 2")

    return [
        DeskTarget(
            global_x=global_x,
            x=current_rail - global_x,
            y=side * table_scan_y,
            z=surface_z,
        )
        for global_x in (
            rail_min + (rail_max - rail_min) * index / (viewpoints - 1)
            for index in range(viewpoints)
        )
    ]


def look_at_eef_rotation(
    *,
    eef_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    eef_to_camera: np.ndarray,
    iterations: int = 4,
) -> np.ndarray:
    """Return base->EEF rotation that aims camera +Z at ``target_position``.

    ``eef_to_camera`` is the homogeneous camera transform relative to the EEF.
    The camera origin changes with the requested EEF rotation, so the look-at
    calculation refines that origin a few times to include its mount offset.
    """
    if eef_to_camera.shape != (4, 4) or not np.all(np.isfinite(eef_to_camera)):
        raise ValueError("eef_to_camera must be a finite 4x4 transform")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    eef = np.asarray(eef_position, dtype=float)
    target = np.asarray(target_position, dtype=float)
    if eef.shape != (3,) or target.shape != (3,) or not np.all(np.isfinite([*eef, *target])):
        raise ValueError("EEF and target positions must be finite XYZ vectors")

    camera_rotation_in_eef = eef_to_camera[:3, :3]
    camera_offset_in_eef = eef_to_camera[:3, 3]
    if not np.allclose(
        camera_rotation_in_eef.T @ camera_rotation_in_eef, np.eye(3), atol=1e-6
    ):
        raise ValueError("eef_to_camera rotation must be orthonormal")

    eef_rotation = np.eye(3)
    global_up = np.array([0.0, 0.0, 1.0])
    for _ in range(iterations):
        camera_position = eef + eef_rotation @ camera_offset_in_eef
        forward = target - camera_position
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-8:
            raise ValueError("camera look-at target coincides with camera origin")
        forward /= forward_norm
        # ROS optical frames use the OpenCV convention: +X right, +Y down,
        # +Z forward. Keep camera +Y toward world-down to avoid an inverted
        # image and the corresponding 180-degree optical-axis twist.
        right = np.cross(forward, global_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1e-8:
            raise ValueError("camera look-at direction is parallel to global up")
        right /= right_norm
        down = np.cross(forward, right)
        camera_rotation = np.column_stack((right, down, forward))
        eef_rotation = camera_rotation @ camera_rotation_in_eef.T

    return eef_rotation


def close_view_pose(
    *, side: int, standoff: float, height: float, roll: float, pitch: float
) -> ScanPose:
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    if standoff <= 0 or height <= 0:
        raise ValueError("stand-off and height must be positive")
    return ScanPose(
        x=0.0,
        y=side * standoff,
        z=height,
        roll=roll,
        pitch=pitch,
        yaw=side * math.pi / 2.0,
    )


def close_view_tilt_poses(
    *,
    side: int,
    standoff: float,
    height: float,
    roll: float,
    pitch: float,
    z_delta: float = 0.05,
    pitch_delta: float = 0.30,
) -> list[ScanPose]:
    """Return a small lower/upper head-tilt pair for close confirmation."""
    if not all(
        math.isfinite(value)
        for value in (standoff, height, roll, pitch, z_delta, pitch_delta)
    ):
        raise ValueError("close-view tilt values must be finite")
    if z_delta <= 0 or pitch_delta <= 0:
        raise ValueError("close-view tilt deltas must be positive")
    if height <= z_delta:
        raise ValueError("close-view lower height must remain positive")

    return [
        close_view_pose(
            side=side,
            standoff=standoff,
            height=height - z_delta,
            roll=roll,
            pitch=pitch - pitch_delta,
        ),
        close_view_pose(
            side=side,
            standoff=standoff,
            height=height + z_delta,
            roll=roll,
            pitch=pitch + pitch_delta,
        ),
    ]
