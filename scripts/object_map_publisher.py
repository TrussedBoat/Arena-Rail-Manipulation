#!/usr/bin/env python3
"""Publish semantic-distance JSON entries as debug TF frames for RViz.

This is intentionally independent from the agent orchestrator.  Run it in a
separate ROS 2 terminal while ``rail_global`` is publishing ``global_origin``.
Each JSON entry becomes a child frame named ``object_<label>`` below the
configured parent frame.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


DEFAULT_JSON = Path(__file__).resolve().parents[1] / "semantic_distances_dynamic.json"
_FRAME_CHARS = re.compile(r"[^A-Za-z0-9_]+")


def frame_name(label: str, prefix: str) -> str:
    """Return a stable TF-safe frame name for a semantic label."""
    cleaned = _FRAME_CHARS.sub("_", label.strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("semantic label produces an empty TF frame name")
    return f"{prefix}{cleaned}"


def read_coordinates(path: Path) -> list[tuple[str, float, float, float]]:
    """Read and validate ``{label: {x, y, z}}`` entries from JSON."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload: Any = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read semantic coordinates {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("semantic coordinate JSON must contain an object")

    coordinates: list[tuple[str, float, float, float]] = []
    for label, value in payload.items():
        if not isinstance(label, str) or not isinstance(value, dict):
            raise RuntimeError(f"invalid semantic coordinate entry for {label!r}")
        try:
            xyz = tuple(float(value[axis]) for axis in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"semantic entry {label!r} must contain numeric x, y, and z"
            ) from exc
        if not all(math.isfinite(component) for component in xyz):
            raise RuntimeError(f"semantic entry {label!r} contains non-finite coordinates")
        coordinates.append((label, *xyz))
    return coordinates


class SemanticDistanceTfDebug(Node):
    def __init__(self, json_path: Path, parent_frame: str, prefix: str, rate_hz: float) -> None:
        super().__init__("object_map_publisher")
        self.json_path = json_path
        self.parent_frame = parent_frame.strip().strip("/")
        self.prefix = prefix
        self._last_mtime_ns: int | None = None
        self._coordinates: list[tuple[str, float, float, float]] = []
        self._broadcaster = TransformBroadcaster(self)
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_if_changed)
        self.get_logger().info(
            f"Publishing semantic TF frames from {self.json_path} "
            f"under {self.parent_frame!r}."
        )
        self._publish_if_changed()

    def _publish_if_changed(self) -> None:
        try:
            mtime_ns = self.json_path.stat().st_mtime_ns
        except OSError as exc:
            self.get_logger().warning(f"Waiting for {self.json_path}: {exc}")
            return
        if mtime_ns != self._last_mtime_ns:
            try:
                self._coordinates = read_coordinates(self.json_path)
            except RuntimeError as exc:
                self.get_logger().error(str(exc))
                return
            self._last_mtime_ns = mtime_ns
            self.get_logger().info(
                f"Loaded {len(self._coordinates)} semantic coordinate(s)."
            )

        stamp = self.get_clock().now().to_msg()
        transforms: list[TransformStamped] = []
        seen_frames: set[str] = set()
        for label, x, y, z in self._coordinates:
            try:
                child_frame = frame_name(label, self.prefix)
            except ValueError as exc:
                self.get_logger().error(f"Skipping invalid label {label!r}: {exc}")
                continue
            if child_frame in seen_frames:
                self.get_logger().error(
                    f"Skipping duplicate TF frame {child_frame!r}; labels collide after sanitizing"
                )
                continue
            seen_frames.add(child_frame)
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.parent_frame
            transform.child_frame_id = child_frame
            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.translation.z = z
            transform.transform.rotation.w = 1.0
            transforms.append(transform)

        for transform in transforms:
            self._broadcaster.sendTransform(transform)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, dest="json_path")
    parser.add_argument("--parent-frame", default="global_origin")
    parser.add_argument("--prefix", default="object_")
    parser.add_argument("--rate", type=float, default=1.0, help="republish rate in Hz")
    args = parser.parse_args()
    if args.rate <= 0 or not math.isfinite(args.rate):
        parser.error("--rate must be a finite positive number")

    rclpy.init()
    node = SemanticDistanceTfDebug(args.json_path, args.parent_frame, args.prefix, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
