"""Geometry helpers for the RRT wrist-camera desk scan."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScanPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


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
