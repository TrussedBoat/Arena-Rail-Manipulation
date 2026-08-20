"""Depth validation, camera deprojection, and rigid transforms."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LocalizedDetection:
    class_name: str
    confidence: float
    class_likelihoods: dict[str, float]
    position: tuple[float, float, float]
    position_covariance: np.ndarray
    stamp_ns: int
    frame_id: str
    bbox_xyxy: tuple[float, float, float, float]
    class_evidence_strength: float = 1.0
    range_m: float = 1.0
    appearance_embedding: np.ndarray | None = None
    appearance_provider_id: str | None = None
    depth_median_m: float = 0.0
    depth_stddev_m: float = 0.0
    depth_valid_pixel_count: int = 0
    depth_inlier_pixel_count: int = 0
    base_pixel_stddev_px: float = 0.0
    bbox_pixel_stddev_px: float = 0.0
    combined_pixel_stddev_px: float = 0.0
    point_cloud: list[dict[str, float | int]] | None = None


@dataclass(frozen=True)
class FixedCameraCalibration:
    """Fixed, rectified camera calibration for an RGB-D stream."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("Camera focal lengths must be positive")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Camera image dimensions must be positive")
        if not 0.0 <= self.cx < self.image_width:
            raise ValueError("Camera cx must lie within the configured image width")
        if not 0.0 <= self.cy < self.image_height:
            raise ValueError("Camera cy must lie within the configured image height")

    @property
    def camera_matrix(self) -> list[float]:
        return [
            self.fx,
            0.0,
            self.cx,
            0.0,
            self.fy,
            self.cy,
            0.0,
            0.0,
            1.0,
        ]


@dataclass(frozen=True)
class DepthSamplingDiagnostics:
    bounds_xyxy: tuple[int, int, int, int]
    valid_mask: np.ndarray
    inlier_mask: np.ndarray
    median_m: float
    stddev_m: float
    centre_pixel: tuple[int, int]


def robust_depth_at_detection(
    depth_m: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    inner_fraction: float,
    minimum_valid_pixels: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> tuple[float, float, tuple[int, int]]:
    diagnostics = depth_sampling_diagnostics(
        depth_m,
        bbox_xyxy,
        inner_fraction,
        minimum_valid_pixels,
        minimum_depth_m,
        maximum_depth_m,
    )
    return diagnostics.median_m, diagnostics.stddev_m, diagnostics.centre_pixel


def depth_sampling_diagnostics(
    depth_m: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    inner_fraction: float,
    minimum_valid_pixels: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> DepthSamplingDiagnostics:
    """Return the exact central ROI and MAD inliers used for depth estimation."""
    if depth_m.ndim != 2:
        raise ValueError("Depth image must be single-channel")
    height, width = depth_m.shape
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid detection bounding box")
    fraction = min(1.0, max(0.05, inner_fraction))
    centre_x = (x1 + x2) / 2.0
    centre_y = (y1 + y2) / 2.0
    half_width = (x2 - x1) * fraction / 2.0
    half_height = (y2 - y1) * fraction / 2.0
    left = max(0, min(width - 1, int(math.floor(centre_x - half_width))))
    right = max(left + 1, min(width, int(math.ceil(centre_x + half_width))))
    top = max(0, min(height - 1, int(math.floor(centre_y - half_height))))
    bottom = max(top + 1, min(height, int(math.ceil(centre_y + half_height))))
    roi = depth_m[top:bottom, left:right]
    valid_mask = (
        np.isfinite(roi)
        & (roi >= minimum_depth_m)
        & (roi <= maximum_depth_m)
    )
    valid = roi[valid_mask]
    if valid.size < minimum_valid_pixels:
        raise ValueError(
            f"Only {valid.size} valid depth pixels; need {minimum_valid_pixels}"
        )
    median = float(np.median(valid))
    absolute_deviation = np.abs(valid - median)
    mad = float(np.median(absolute_deviation))
    inlier_mask = valid_mask.copy()
    if mad > 0.0:
        valid_inliers = absolute_deviation <= 3.5 * 1.4826 * mad
        inlier_mask[valid_mask] = valid_inliers
        inliers = valid[valid_inliers]
        if inliers.size >= minimum_valid_pixels:
            median = float(np.median(inliers))
            mad = float(np.median(np.abs(inliers - median)))
    stddev = max(0.005, 1.4826 * mad)
    pixel = (
        max(0, min(width - 1, int(round(centre_x)))),
        max(0, min(height - 1, int(round(centre_y)))),
    )
    return DepthSamplingDiagnostics(
        bounds_xyxy=(left, top, right, bottom),
        valid_mask=valid_mask,
        inlier_mask=inlier_mask,
        median_m=median,
        stddev_m=stddev,
        centre_pixel=pixel,
    )


def deproject_pixel(
    pixel: tuple[int, int], depth_m: float, camera_matrix: list[float]
) -> np.ndarray:
    if len(camera_matrix) != 9:
        raise ValueError("Camera intrinsics must contain nine values")
    fx, fy = float(camera_matrix[0]), float(camera_matrix[4])
    cx, cy = float(camera_matrix[2]), float(camera_matrix[5])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera focal lengths must be positive")
    u, v = pixel
    return np.array(
        [(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m],
        dtype=np.float64,
    )


def deprojection_covariance(
    pixel: tuple[int, int],
    depth_m: float,
    depth_stddev_m: float,
    camera_matrix: list[float],
    pixel_stddev_px: float,
) -> np.ndarray:
    """Propagate independent pixel/depth uncertainty into the camera frame."""
    if len(camera_matrix) != 9:
        raise ValueError("Camera intrinsics must contain nine values")
    fx, fy = float(camera_matrix[0]), float(camera_matrix[4])
    cx, cy = float(camera_matrix[2]), float(camera_matrix[5])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera focal lengths must be positive")
    if depth_stddev_m <= 0.0 or pixel_stddev_px <= 0.0:
        raise ValueError("Measurement standard deviations must be positive")
    u, v = pixel
    jacobian = np.array(
        [
            [depth_m / fx, 0.0, (u - cx) / fx],
            [0.0, depth_m / fy, (v - cy) / fy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    source_covariance = np.diag(
        [pixel_stddev_px**2, pixel_stddev_px**2, depth_stddev_m**2]
    )
    return jacobian @ source_covariance @ jacobian.T


def rotation_matrix_from_transform(transform: object) -> np.ndarray:
    q = transform.rotation
    quaternion = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("TF quaternion has zero norm")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_covariance(
    covariance: np.ndarray,
    transform: object,
    extrinsic_stddev_m: float,
    world_stddev_m: float,
) -> np.ndarray:
    """Rotate covariance into the target frame and add calibration/TF floors."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("Position covariance must be a finite 3x3 matrix")
    if extrinsic_stddev_m < 0.0 or world_stddev_m < 0.0:
        raise ValueError("Covariance floor standard deviations cannot be negative")
    rotation = rotation_matrix_from_transform(transform)
    floor_variance = extrinsic_stddev_m**2 + world_stddev_m**2
    result = rotation @ covariance @ rotation.T
    result += np.eye(3, dtype=np.float64) * floor_variance
    return (result + result.T) / 2.0


def transform_point(point: np.ndarray, transform: object) -> np.ndarray:
    translation = transform.translation
    rotation = rotation_matrix_from_transform(transform)
    return rotation @ point + np.array(
        [translation.x, translation.y, translation.z], dtype=np.float64
    )
