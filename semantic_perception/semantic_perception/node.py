"""ROS 2 RGB-D semantic perception node."""

from dataclasses import dataclass
from collections import deque
from pathlib import Path
import math
import threading
import time

import cv2
import numpy as np

import message_filters
import rclpy
from interface.action import FindObject
from interface.msg import (
    AssociationDiagnostic,
    AssociationDiagnosticArray,
    ClassProbability,
    DetectedObject,
    DetectedObjectArray,
    SemanticTextMatch,
)
from interface.srv import SearchSemanticObjects
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import Image
import tf2_ros
from tf2_ros import TransformException

from .detector import ObjectDetector, create_detector
from .appearance import create_appearance_provider
from .localization import (
    LocalizedDetection,
    DepthSamplingDiagnostics,
    depth_sampling_diagnostics,
    FixedCameraCalibration,
    deproject_pixel,
    deprojection_covariance,
    robust_depth_at_detection,
    transform_covariance,
    transform_point,
)
from .registry import AssociationResult, ObjectRegistry, ObjectTrack, RegistryConfig


@dataclass(frozen=True)
class SensorPacket:
    sequence: int
    color: Image
    depth: Image
    received_monotonic: float


@dataclass(frozen=True)
class RejectedDepthDetection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    reason: str


@dataclass(frozen=True)
class RejectedImageDetection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    reason: str


class SemanticPerceptionNode(Node):
    def __init__(self, detector: ObjectDetector | None = None) -> None:
        super().__init__("semantic_perception")
        self._declare_parameters()
        self._validate_parameters()
        self._packet_lock = threading.Lock()
        self._registry_lock = threading.RLock()
        self._registry_changed = threading.Condition(self._registry_lock)
        self._pending_packets: deque[SensorPacket] = deque(
            maxlen=int(self.get_parameter("tf.sync_queue_size").value)
        )
        self._received_sequence = 0
        self._processed_sequence = 0
        self._processed_packets_count = 0
        self._color_messages_received = 0
        self._depth_messages_received = 0
        self._synchronized_packets_received = 0
        self._rgb_depth_skew_rejected = 0
        self._last_health_processed = 0
        self._tf_wait_expirations = 0
        self._last_health_monotonic = time.monotonic()
        self._last_timing_monotonic = time.monotonic()
        self._last_persist_monotonic = 0.0
        self._registry_dirty = False
        self._camera_calibration = self._camera_calibration_from_parameters()

        self._detector = detector or create_detector(
            backend=str(self.get_parameter("detector.backend").value),
            model_path=str(self.get_parameter("detector.model_path").value),
            device=str(self.get_parameter("detector.device").value),
            image_size=int(self.get_parameter("detector.image_size").value),
            confidence_threshold=float(
                self.get_parameter("detector.confidence_threshold").value
            ),
            max_detections=int(self.get_parameter("detector.max_detections").value),
            class_reliability_floor=float(
                self.get_parameter("class.conditional_reliability_floor").value
            ),
        )
        self._appearance_provider = create_appearance_provider(
            enabled=bool(self.get_parameter("appearance.enabled").value),
            backend=str(self.get_parameter("appearance.backend").value),
            detector=self._detector,
            device=str(self.get_parameter("detector.device").value),
            image_size=int(self.get_parameter("appearance.image_size").value),
            mobileclip_checkpoint=str(
                self.get_parameter("appearance.mobileclip_checkpoint").value
            ),
        )
        self._registry = ObjectRegistry(
            RegistryConfig(
                mahalanobis_threshold=float(
                    self.get_parameter("association.mahalanobis_threshold").value
                ),
                confirmation_hits=int(self.get_parameter("filter.confirmation_hits").value),
                confirmation_window_sec=float(
                    self.get_parameter("filter.confirmation_window_sec").value
                ),
                class_confirmation_probability=float(
                    self.get_parameter("class.confirmation_probability").value
                ),
                max_explicit_classes=int(
                    self.get_parameter("class.max_explicit_classes").value
                ),
                stale_after_sec=float(self.get_parameter("filter.stale_after_sec").value),
                stale_retention_sec=float(
                    self.get_parameter("filter.stale_retention_sec").value
                ),
                evidence_decay=float(self.get_parameter("class.evidence_decay").value),
                process_noise_stddev_m=float(
                    self.get_parameter("uncertainty.process_noise_stddev_m").value
                ),
                appearance_max_cosine_distance=float(
                    self.get_parameter("appearance.max_cosine_distance").value
                ),
                appearance_cost_weight=float(
                    self.get_parameter("appearance.cost_weight").value
                ),
                appearance_embedding_decay=float(
                    self.get_parameter("appearance.embedding_decay").value
                ),
                distance_evidence_reference_m=float(
                    self.get_parameter("distance_weighting.reference_distance_m").value
                ),
                distance_evidence_min_weight=float(
                    self.get_parameter("distance_weighting.minimum_class_evidence_weight").value
                ),
            ),
            model_id=self._detector.model_id,
        )
        self._registry_path = Path(str(self.get_parameter("registry.path").value))
        self._legacy_path = Path(
            str(self.get_parameter("registry.legacy_coordinates_path").value)
        )
        self._load_registry()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._detections_publisher = self.create_publisher(
            DetectedObjectArray, "/semantic/detections", 10
        )
        registry_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._objects_publisher = self.create_publisher(
            DetectedObjectArray, "/semantic/objects", registry_qos
        )
        self._target_found_publisher = self.create_publisher(
            DetectedObject, "/semantic/target_found", 10
        )
        self._debug_image_publisher = self.create_publisher(
            Image, str(self.get_parameter("debug.annotated_topic").value), 10
        )
        self._association_diagnostics_publisher = self.create_publisher(
            AssociationDiagnosticArray, "/semantic/association_diagnostics", 10
        )

        color_topic = str(self.get_parameter("topics.color").value)
        depth_topic = str(self.get_parameter("topics.aligned_depth").value)
        self._color_subscription = message_filters.Subscriber(
            self, Image, color_topic, qos_profile=qos_profile_sensor_data
        )
        self._depth_subscription = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=qos_profile_sensor_data
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._color_subscription, self._depth_subscription],
            queue_size=int(self.get_parameter("sync.queue_size").value),
            slop=float(self.get_parameter("sync.slop_sec").value),
        )
        self._color_subscription.registerCallback(self._color_received)
        self._depth_subscription.registerCallback(self._depth_received)
        self._synchronizer.registerCallback(self._sensor_callback)

        rate_hz = float(self.get_parameter("processing_rate_hz").value)
        self._processing_timer = self.create_timer(1.0 / rate_hz, self._process_latest)
        self._persistence_timer = self.create_timer(1.0, self._persist_if_due)
        self._lifecycle_timer = self.create_timer(1.0, self._maintain_registry_lifecycle)
        self._input_health_timer = self.create_timer(1.0, self._log_input_health)
        # Voxel hash tracking for 3DGS camera poses
        self._saved_voxels: dict[str, set[tuple[int, int, int, int, int, int]]] = {}

        self._action_server = ActionServer(
            self,
            FindObject,
            "/semantic/find_object",
            execute_callback=self._execute_find_object,
            goal_callback=self._find_object_goal_callback,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self._text_search_service = self.create_service(
            SearchSemanticObjects,
            "/semantic/search_objects",
            self._search_objects_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self._publish_registry()
        self.get_logger().info(
            f"Semantic perception ready: model={self._detector.model_id}, "
            f"appearance={self._appearance_provider.provider_id}, "
            f"frame={self._reference_frame()}, rate={rate_hz:g} Hz"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("topics.color", "/sim/rail_franka1/cam/wrist/color/image_raw")
        self.declare_parameter(
            "topics.aligned_depth", "/sim/rail_franka1/cam/wrist/depth/image_raw"
        )
        self.declare_parameter("camera.fx", 907.00)
        self.declare_parameter("camera.fy", 905.69)
        self.declare_parameter("camera.cx", 567.74)
        self.declare_parameter("camera.cy", 488.32)
        self.declare_parameter("camera.image_width", 1280)
        self.declare_parameter("camera.image_height", 720)
        self.declare_parameter("camera.tf_frame_override", "wrist_camera")
        self.declare_parameter("reference_frame", "global_origin")
        self.declare_parameter("processing_rate_hz", 5.0)
        self.declare_parameter("sync.queue_size", 8)
        self.declare_parameter("sync.slop_sec", 0.03)
        self.declare_parameter("sync.max_rgb_depth_delta_sec", 0.015)
        self.declare_parameter("tf_timeout_sec", 0.2)
        self.declare_parameter("tf.sync_queue_size", 30)
        self.declare_parameter("tf.max_wait_sec", 0.5)
        self.declare_parameter("tf.post_image_guard_sec", 0.03)
        self.declare_parameter("tf.fallback_to_latest", False)
        self.declare_parameter("detector.backend", "ultralytics")
        self.declare_parameter("detector.model_path", "")
        self.declare_parameter("detector.device", "cuda:0")
        self.declare_parameter("detector.image_size", 640)
        self.declare_parameter("detector.confidence_threshold", 0.50)
        self.declare_parameter("detector.max_detections", 100)
        self.declare_parameter("detector.max_bbox_area_fraction", 0.55)
        self.declare_parameter("detector.border_margin_fraction", 0.03)
        self.declare_parameter("appearance.enabled", True)
        self.declare_parameter("appearance.backend", "mobileclip_s0")
        self.declare_parameter("appearance.mobileclip_checkpoint", "")
        self.declare_parameter("appearance.image_size", 224)
        self.declare_parameter("appearance.max_cosine_distance", 0.35)
        self.declare_parameter("appearance.cost_weight", 0.5)
        self.declare_parameter("appearance.embedding_decay", 0.90)
        self.declare_parameter("distance_weighting.reference_distance_m", 1.0)
        self.declare_parameter("distance_weighting.minimum_class_evidence_weight", 0.25)
        self.declare_parameter("depth.scale_16uc1", 0.001)
        self.declare_parameter("depth.inner_bbox_fraction", 0.5)
        self.declare_parameter("depth.minimum_valid_pixels", 20)
        self.declare_parameter("depth.minimum_m", 0.05)
        self.declare_parameter("depth.maximum_m", 5.0)
        self.declare_parameter("association.mahalanobis_threshold", 11.345)
        self.declare_parameter("association.duplicate_merge_enabled", True)
        self.declare_parameter("association.duplicate_merge_interval_frames", 25)
        self.declare_parameter("filter.confirmation_hits", 3)
        self.declare_parameter("filter.confirmation_window_sec", 3.0)
        self.declare_parameter("filter.stale_after_sec", 300.0)
        self.declare_parameter("filter.stale_retention_sec", 120.0)
        self.declare_parameter("class.confirmation_probability", 0.70)
        self.declare_parameter("class.conditional_reliability_floor", 0.60)
        self.declare_parameter("class.max_explicit_classes", 4)
        self.declare_parameter("class.evidence_decay", 0.95)
        self.declare_parameter("uncertainty.pixel_stddev_px", 2.0)
        self.declare_parameter("uncertainty.bbox_diagonal_fraction", 0.10)
        self.declare_parameter("uncertainty.extrinsic_stddev_m", 0.005)
        self.declare_parameter("uncertainty.world_stddev_m", 0.005)
        self.declare_parameter("uncertainty.process_noise_stddev_m", 0.002)
        self.declare_parameter("registry.path", "semantic_objects.json")
        self.declare_parameter(
            "registry.legacy_coordinates_path", "semantic_distances_dynamic.json"
        )
        self.declare_parameter("registry.write_legacy_coordinates", False)
        self.declare_parameter("registry.persistence_interval_sec", 1.0)
        self.declare_parameter("search.default_timeout_sec", 30.0)
        self.declare_parameter("semantic_text.enabled", True)
        self.declare_parameter("semantic_text.minimum_cosine_similarity", 0.25)
        self.declare_parameter("semantic_text.max_results", 5)
        self.declare_parameter("debug.publish_annotated", False)
        self.declare_parameter("debug.timing", False)
        self.declare_parameter("debug.timing_log_interval_sec", 1.0)
        self.declare_parameter("debug.annotated_topic", "/semantic/debug/yolo")
        self.declare_parameter("diagnostics.publish", True)
        
        # 3DGS Parameters
        self.declare_parameter("export.voxel_size_m", 0.05)
        self.declare_parameter("export.voxel_size_deg", 5.0)

    def _validate_parameters(self) -> None:
        positive_parameters = (
            "processing_rate_hz",
            "sync.queue_size",
            "sync.slop_sec",
            "sync.max_rgb_depth_delta_sec",
            "tf_timeout_sec",
            "tf.sync_queue_size",
            "tf.max_wait_sec",
            "tf.post_image_guard_sec",
            "detector.image_size",
            "detector.max_detections",
            "appearance.image_size",
            "distance_weighting.reference_distance_m",
            "depth.minimum_valid_pixels",
            "depth.maximum_m",
            "association.mahalanobis_threshold",
            "association.duplicate_merge_interval_frames",
            "filter.confirmation_hits",
            "filter.confirmation_window_sec",
            "filter.stale_after_sec",
            "filter.stale_retention_sec",
            "class.max_explicit_classes",
            "uncertainty.pixel_stddev_px",
            "uncertainty.process_noise_stddev_m",
            "registry.persistence_interval_sec",
            "search.default_timeout_sec",
            "semantic_text.max_results",
        )
        for name in positive_parameters:
            if float(self.get_parameter(name).value) <= 0.0:
                raise ValueError(f"ROS parameter {name} must be positive")
        unit_parameters = (
            "detector.confidence_threshold",
            "detector.max_bbox_area_fraction",
            "detector.border_margin_fraction",
            "appearance.max_cosine_distance",
            "appearance.cost_weight",
            "appearance.embedding_decay",
            "uncertainty.bbox_diagonal_fraction",
            "distance_weighting.minimum_class_evidence_weight",
            "depth.inner_bbox_fraction",
            "class.confirmation_probability",
            "class.conditional_reliability_floor",
            "class.evidence_decay",
            "semantic_text.minimum_cosine_similarity",
        )
        for name in unit_parameters:
            value = float(self.get_parameter(name).value)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"ROS parameter {name} must be in (0, 1]")
        for name in (
            "uncertainty.extrinsic_stddev_m",
            "uncertainty.world_stddev_m",
        ):
            if float(self.get_parameter(name).value) < 0.0:
                raise ValueError(f"ROS parameter {name} cannot be negative")
        self._camera_calibration_from_parameters()
        if not str(self.get_parameter("reference_frame").value).strip():
            raise ValueError("reference_frame must not be empty")
        if float(self.get_parameter("depth.maximum_m").value) <= float(
            self.get_parameter("depth.minimum_m").value
        ):
            raise ValueError("depth.maximum_m must be greater than depth.minimum_m")

    def _sensor_callback(self, color: Image, depth: Image) -> None:
        rgb_depth_delta_sec = abs(
            Time.from_msg(color.header.stamp).nanoseconds
            - Time.from_msg(depth.header.stamp).nanoseconds
        ) / 1e9
        if rgb_depth_delta_sec > float(
            self.get_parameter("sync.max_rgb_depth_delta_sec").value
        ):
            self._rgb_depth_skew_rejected += 1
            self.get_logger().debug(
                "Rejected RGB-D pair with timestamp delta %.3f ms" % (
                    rgb_depth_delta_sec * 1e3
                ),
                throttle_duration_sec=1.0,
            )
            return
        with self._packet_lock:
            self._received_sequence += 1
            self._synchronized_packets_received += 1
            self._pending_packets.append(SensorPacket(
                sequence=self._received_sequence,
                color=color,
                depth=depth,
                received_monotonic=time.monotonic(),
            ))

    def _color_received(self, _: Image) -> None:
        self._color_messages_received += 1

    def _depth_received(self, _: Image) -> None:
        self._depth_messages_received += 1

    def _packet_source_frame(self, packet: SensorPacket) -> str:
        source_frame = str(
            self.get_parameter("camera.tf_frame_override").value
        ).strip()
        if not source_frame:
            source_frame = packet.color.header.frame_id.strip()
        if not source_frame:
            raise ValueError("RGB image has no frame_id")
        return source_frame

    def _full_crop_splat_points(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        bounds_xyxy: tuple[int, int, int, int],
        camera_to_world: np.ndarray,
    ) -> list[dict[str, float | int]]:
        """Project every valid-depth pixel in a YOLO crop for 3DGS seeding.

        This intentionally retains object surroundings and background context.
        It is separate from the compact, object-depth-filtered tracking cloud.
        """
        left, top, right, bottom = bounds_xyxy
        crop_depth = depth[top:bottom, left:right]
        valid = (
            np.isfinite(crop_depth)
            & (crop_depth >= float(self.get_parameter("depth.minimum_m").value))
            & (crop_depth <= float(self.get_parameter("depth.maximum_m").value))
        )
        crop_v, crop_u = np.nonzero(valid)
        if crop_u.size == 0:
            return []

        z = crop_depth[crop_v, crop_u].astype(np.float64, copy=False)
        u = crop_u.astype(np.float64) + left
        v = crop_v.astype(np.float64) + top
        calibration = self._camera_calibration
        camera_points = np.column_stack((
            (u - calibration.cx) * z / calibration.fx,
            (v - calibration.cy) * z / calibration.fy,
            z,
        ))
        world_points = (
            camera_points @ camera_to_world[:3, :3].T
            + camera_to_world[:3, 3]
        )
        # OpenCV/ROS color images are BGR; cache RGB for PLY initialization.
        bgr = color[top + crop_v, left + crop_u]
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

    def _tf_ready_for_packet(self, packet: SensorPacket) -> bool:
        """Whether TF brackets this camera timestamp with the configured guard."""
        source_frame = self._packet_source_frame(packet)
        image_stamp = Time.from_msg(packet.color.header.stamp)
        if not self._tf_buffer.can_transform(
            self._reference_frame(),
            source_frame,
            image_stamp,
            timeout=Duration(seconds=0.0),
        ):
            return False
        guard_sec = float(self.get_parameter("tf.post_image_guard_sec").value)
        if guard_sec <= 0.0:
            return True
        latest_transform = self._tf_buffer.lookup_transform(
            self._reference_frame(), source_frame, Time(), timeout=Duration(seconds=0.0)
        )
        latest_stamp = Time.from_msg(latest_transform.header.stamp)
        return latest_stamp.nanoseconds - image_stamp.nanoseconds >= int(guard_sec * 1e9)

    def _log_input_health(self) -> None:
        now_monotonic = time.monotonic()
        elapsed_sec = max(now_monotonic - self._last_health_monotonic, 1e-6)
        processed_hz = (
            self._processed_packets_count - self._last_health_processed
        ) / elapsed_sec
        confirmation_window_sec = float(
            self.get_parameter("filter.confirmation_window_sec").value
        )
        confirmation_hits = int(self.get_parameter("filter.confirmation_hits").value)
        available_observations = processed_hz * confirmation_window_sec
        confirmation_ratio = confirmation_hits / max(available_observations, 1e-6)
        with self._packet_lock:
            pending_tf_packets = len(self._pending_packets)
        self.get_logger().info(
            "RGB-D input health: color=%d depth=%d synchronized=%d processed=%d "
            "processed_hz=%.2f confirmation=%d/%.1f(%.0f%%) "
            "rgb_depth_skew_rejected=%d tf_pending=%d tf_wait_expired=%d"
            % (
                self._color_messages_received,
                self._depth_messages_received,
                self._synchronized_packets_received,
                self._processed_packets_count,
                processed_hz,
                confirmation_hits,
                available_observations,
                confirmation_ratio * 100.0,
                self._rgb_depth_skew_rejected,
                pending_tf_packets,
                self._tf_wait_expirations,
            )
        )
        self._last_health_processed = self._processed_packets_count
        self._last_health_monotonic = now_monotonic

    def _maintain_registry_lifecycle(self) -> None:
        """Publish/persist stale transitions even while no new camera packet arrives."""
        now_sec = self.get_clock().now().nanoseconds / 1e9
        with self._registry_changed:
            expired_candidates, marked_stale, deleted_stale = self._registry.maintain(now_sec)
            if not expired_candidates and not marked_stale and not deleted_stale:
                return
            self._registry_dirty = True
            self._registry_changed.notify_all()
        self._publish_registry()
        if expired_candidates:
            self.get_logger().debug("Expired candidate tracks: %s" % expired_candidates)
        if marked_stale:
            self.get_logger().info("Marked stale tracks: %s" % marked_stale)
        if deleted_stale:
            self.get_logger().info("Deleted stale tracks: %s" % deleted_stale)

    def _process_latest(self) -> None:
        packet: SensorPacket | None = None
        with self._packet_lock:
            pending = [
                item for item in self._pending_packets
                if item.sequence > self._processed_sequence
            ]
        if not pending:
            return

        # Prefer the newest RGB-D pair for which TF can interpolate at the
        # image timestamp. This deliberately lets a lower-rate TF stream lag
        # behind the camera instead of applying a transform from another time.
        for candidate in reversed(pending):
            try:
                transform_ready = self._tf_ready_for_packet(candidate)
            except (TransformException, ValueError):
                transform_ready = False
            if transform_ready:
                packet = candidate
                break

        if packet is None:
            oldest = pending[0]
            waited_sec = time.monotonic() - oldest.received_monotonic
            if waited_sec < float(self.get_parameter("tf.max_wait_sec").value):
                return
            # Once the bounded wait expires, process the oldest pair. Exact TF
            # lookup will either reject it or use the explicitly enabled
            # latest-transform fallback.
            packet = oldest
            self._tf_wait_expirations += 1

        with self._packet_lock:
            if packet.sequence <= self._processed_sequence:
                return
            self._processed_sequence = packet.sequence
            self._processed_packets_count += 1
            while (
                self._pending_packets
                and self._pending_packets[0].sequence <= packet.sequence
            ):
                self._pending_packets.popleft()
        frame_started = time.perf_counter()
        try:
            localized, rejected_depth, tf_mode, selected_tf_stamp, timing = self._localize_packet(packet)
        except Exception as exc:
            self.get_logger().warning(f"Dropped RGB-D packet: {exc}")
            return
        association_started = time.perf_counter()
        self._publish_observations(localized, packet.color)
        association_results: list[AssociationResult] = []
        changed_tracks: list[ObjectTrack] = []
        merged_duplicates: list[tuple[str, str]] = []
        expired_candidates: list[str] = []
        deleted_stale: list[str] = []
        with self._registry_changed:
            expired_candidates, marked_stale, deleted_stale = self._registry.maintain(
                self.get_clock().now().nanoseconds / 1e9
            )
            association_results = self._registry.update_frame_with_diagnostics(localized)
            changed_tracks = [result.track for result in association_results]
            merge_interval = int(
                self.get_parameter("association.duplicate_merge_interval_frames").value
            )
            if (
                bool(self.get_parameter("association.duplicate_merge_enabled").value)
                and merge_interval > 0
                and self._processed_sequence % merge_interval == 0
            ):
                merged_duplicates = self._registry.merge_confirmed_duplicates()
            if changed_tracks or expired_candidates or marked_stale or deleted_stale:
                self._registry_dirty = True
                self._registry_changed.notify_all()
            if merged_duplicates:
                self._registry_dirty = True
                self._registry_changed.notify_all()
        if changed_tracks or merged_duplicates or expired_candidates or marked_stale or deleted_stale:
            self._publish_registry()
        if expired_candidates:
            self.get_logger().debug("Expired candidate tracks: %s" % expired_candidates)
        if marked_stale:
            self.get_logger().info("Marked stale tracks: %s" % marked_stale)
        if deleted_stale:
            self.get_logger().info("Deleted stale tracks: %s" % deleted_stale)
        if merged_duplicates:
            self.get_logger().info("Merged duplicate tracks: %s" % merged_duplicates)
        timing["association"] = time.perf_counter() - association_started
        diagnostics_started = time.perf_counter()
        if bool(self.get_parameter("diagnostics.publish").value):
            self._publish_association_diagnostics(
                packet.color,
                localized,
                association_results,
                rejected_depth,
                tf_mode,
                selected_tf_stamp,
            )
        timing["diagnostics"] = time.perf_counter() - diagnostics_started
        persistence_started = time.perf_counter()
        self._persist_if_due()
        timing["persistence"] = time.perf_counter() - persistence_started
        # Export 3DGS Data
        import os
        import json
        import cv2
        for result in association_results:
            if (
                result.decision != "unmatched" 
                and result.track is not None 
                and result.track.state == "confirmed"
            ):
                det = localized[result.detection_index]
                if det.color_crop is not None and det.camera_to_world is not None:
                    
                    # 6D Voxel Hashing Filter
                    voxel_size_m = float(self.get_parameter("export.voxel_size_m").value)
                    voxel_size_rad = float(self.get_parameter("export.voxel_size_deg").value) * math.pi / 180.0
                    
                    tx, ty, tz = det.camera_to_world[0, 3], det.camera_to_world[1, 3], det.camera_to_world[2, 3]
                    R = det.camera_to_world[:3, :3]
                    yaw = np.arctan2(R[1, 0], R[0, 0])
                    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
                    roll = np.arctan2(R[2, 1], R[2, 2])
                    
                    voxel_hash = (
                        int(math.floor(tx / voxel_size_m)),
                        int(math.floor(ty / voxel_size_m)),
                        int(math.floor(tz / voxel_size_m)),
                        int(math.floor(roll / voxel_size_rad)),
                        int(math.floor(pitch / voxel_size_rad)),
                        int(math.floor(yaw / voxel_size_rad))
                    )
                    
                    object_id = result.track.object_id
                    obj_dir = f"/dev/shm/3dgs_cache/{object_id}"
                    voxels_path = os.path.join(obj_dir, "voxels.json")
                    
                    if object_id not in self._saved_voxels:
                        self._saved_voxels[object_id] = set()
                        # Attempt to load existing voxels if node restarted
                        if os.path.isfile(voxels_path):
                            try:
                                with open(voxels_path, "r") as f:
                                    loaded_voxels = json.load(f)
                                    self._saved_voxels[object_id] = {tuple(v) for v in loaded_voxels}
                            except Exception as e:
                                self.get_logger().warning(f"Failed to load voxels for {object_id}: {e}")
                        
                    if voxel_hash in self._saved_voxels[object_id]:
                        continue  # Skip this frame; voxel already filled!
                        
                    self._saved_voxels[object_id].add(voxel_hash)
                    
                    os.makedirs(obj_dir, exist_ok=True)
                    
                    # Save updated voxels to disk
                    try:
                        with open(voxels_path, "w") as f:
                            json.dump([list(v) for v in self._saved_voxels[object_id]], f)
                    except Exception as e:
                        self.get_logger().warning(f"Failed to save voxels for {object_id}: {e}")
                    
                    rgb_crop = det.color_crop.copy()
                    
                    stamp_ns = det.stamp_ns
                    img_path = os.path.join(obj_dir, f"{stamp_ns}.png")
                    cv2.imwrite(img_path, rgb_crop)
                    
                    # Save depth crop as uint16 millimeters
                    if det.depth_crop is not None:
                        depth_mm = np.clip(det.depth_crop * 1000.0, 0, 65535).astype(np.uint16)
                        depth_path = os.path.join(obj_dir, f"{stamp_ns}_depth.png")
                        cv2.imwrite(depth_path, depth_mm)
                    
                    cx = self._camera_calibration.cx - det.crop_offset[0]
                    cy = self._camera_calibration.cy - det.crop_offset[1]
                    
                    meta = {
                        "camera_to_world": det.camera_to_world.tolist(),
                        "intrinsics": {
                            "fx": self._camera_calibration.fx,
                            "fy": self._camera_calibration.fy,
                            "cx": cx,
                            "cy": cy,
                            "width": int(rgb_crop.shape[1]),
                            "height": int(rgb_crop.shape[0])
                        },
                        "point_cloud": (
                            det.splat_point_cloud
                            if det.splat_point_cloud is not None else []
                        ),
                    }
                    meta_path = os.path.join(obj_dir, f"{stamp_ns}.json")
                    with open(meta_path, "w") as f:
                        json.dump(meta, f)

        timing["total"] = time.perf_counter() - frame_started
        now_monotonic = time.monotonic()
        if (
            bool(self.get_parameter("debug.timing").value)
            and now_monotonic - self._last_timing_monotonic
            >= float(self.get_parameter("debug.timing_log_interval_sec").value)
        ):
            self._last_timing_monotonic = now_monotonic
            self.get_logger().debug(
                "Timing ms: total={total:.1f} decode={decode:.1f} tf={tf:.1f} "
                "yolo={yolo:.1f} filter={filter:.1f} appearance={appearance:.1f} "
                "depth={depth:.1f} debug_image={debug_image:.1f} association={association:.1f} diagnostics={diagnostics:.1f} "
                "persist={persistence:.1f} detections={detections} localized={localized}".format(
                    **{
                        **{name: seconds * 1000.0 for name, seconds in timing.items()},
                        "detections": timing["detections"],
                        "localized": len(localized),
                    }
                )
            )

    def _localize_packet(
        self, packet: SensorPacket
    ) -> tuple[list[LocalizedDetection], list[RejectedDepthDetection], str, Time, dict[str, float | int]]:
        timing: dict[str, float | int] = {}
        stage_started = time.perf_counter()
        color = _decode_color_image(packet.color)
        depth = _decode_depth_image(packet.depth)
        if packet.depth.encoding.upper() in ("16UC1", "MONO16"):
            depth *= float(self.get_parameter("depth.scale_16uc1").value)
        if color.shape[:2] != depth.shape[:2]:
            raise ValueError(
                f"RGB {color.shape[:2]} and depth {depth.shape[:2]} are not aligned"
            )
        expected_shape = (
            self._camera_calibration.image_height,
            self._camera_calibration.image_width,
        )
        if color.shape[:2] != expected_shape:
            raise ValueError(
                f"RGB dimensions {color.shape[:2]} do not match fixed calibration "
                f"{expected_shape}"
            )
        timing["decode"] = time.perf_counter() - stage_started
        source_frame = self._packet_source_frame(packet)
        stamp = Time.from_msg(packet.color.header.stamp)
        stage_started = time.perf_counter()
        try:
            transform = self._tf_buffer.lookup_transform(
                self._reference_frame(),
                source_frame,
                stamp,
                timeout=Duration(
                    seconds=float(self.get_parameter("tf_timeout_sec").value)
                ),
            )
            tf_mode = "exact"
        except TransformException as exc:
            if not bool(self.get_parameter("tf.fallback_to_latest").value):
                raise
            self.get_logger().warning(
                "Exact TF lookup failed (%s); using latest transform because "
                "camera and TF timestamps use different time domains." % exc,
                throttle_duration_sec=5.0,
            )
            transform = self._tf_buffer.lookup_transform(
                self._reference_frame(), source_frame, Time()
            )
            tf_mode = "latest_fallback"
        timing["tf"] = time.perf_counter() - stage_started
        # Registry time must use one clock domain.  Image header stamps are
        # retained for TF lookup, but simulators may stamp images with wall
        # time while this node runs from /clock.  Mixing those domains makes a
        # new track appear immediately old (or infinitely fresh).
        registry_stamp_ns = self.get_clock().now().nanoseconds
        if bool(self.get_parameter("use_sim_time").value) and abs(
            stamp.nanoseconds - registry_stamp_ns
        ) > 5_000_000_000:
            self.get_logger().warning(
                "Camera and registry clock domains differ: image=%d registry=%d; "
                "using registry clock for track lifecycle." % (
                    stamp.nanoseconds,
                    registry_stamp_ns,
                ),
                throttle_duration_sec=5.0,
            )
        stage_started = time.perf_counter()
        detector_output = self._detector.detect(color)
        timing["yolo"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        raw_detections, rejected_image_detections = self._filter_image_detections(
            detector_output, color.shape[:2]
        )
        timing["filter"] = time.perf_counter() - stage_started
        timing["detections"] = len(raw_detections)
        stage_started = time.perf_counter()
        try:
            raw_detections = self._appearance_provider.attach(color, raw_detections)
        except RuntimeError as exc:
            self.get_logger().warning(
                f"Appearance embeddings unavailable; using spatial association only: {exc}",
                throttle_duration_sec=5.0,
            )
        timing["appearance"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        localized: list[LocalizedDetection] = []
        depth_diagnostics: dict[int, DepthSamplingDiagnostics] = {}
        rejected_depth: list[RejectedDepthDetection] = []
        for detection_index, detection in enumerate(raw_detections):
            try:
                diagnostics = depth_sampling_diagnostics(
                    depth,
                    detection.bbox_xyxy,
                    inner_fraction=float(
                        self.get_parameter("depth.inner_bbox_fraction").value
                    ),
                    minimum_valid_pixels=int(
                        self.get_parameter("depth.minimum_valid_pixels").value
                    ),
                    minimum_depth_m=float(self.get_parameter("depth.minimum_m").value),
                    maximum_depth_m=float(self.get_parameter("depth.maximum_m").value),
                )
                depth_diagnostics[detection_index] = diagnostics
                depth_m = diagnostics.median_m
                depth_stddev = diagnostics.stddev_m
                # Adaptive depth threshold for background removal (MAD-based)
                adaptive_depth_threshold = float(np.clip(
                    3.0 * depth_stddev, 0.03, 0.25
                ))
                pixel = diagnostics.centre_pixel
                camera_point = deproject_pixel(
                    pixel, depth_m, self._camera_calibration.camera_matrix
                )
                box_width_px = max(0.0, detection.bbox_xyxy[2] - detection.bbox_xyxy[0])
                box_height_px = max(0.0, detection.bbox_xyxy[3] - detection.bbox_xyxy[1])
                base_pixel_stddev_px = float(self.get_parameter("uncertainty.pixel_stddev_px").value)
                bbox_pixel_stddev_px = float(
                    self.get_parameter("uncertainty.bbox_diagonal_fraction").value
                ) * float(np.hypot(box_width_px, box_height_px))
                combined_pixel_stddev_px = float(
                    np.hypot(base_pixel_stddev_px, bbox_pixel_stddev_px)
                )
                camera_covariance = deprojection_covariance(
                    pixel,
                    depth_m,
                    depth_stddev,
                    self._camera_calibration.camera_matrix,
                    pixel_stddev_px=combined_pixel_stddev_px,
                )
                world_point = transform_point(camera_point, transform.transform)
                world_covariance = transform_covariance(
                    camera_covariance,
                    transform.transform,
                    extrinsic_stddev_m=float(
                        self.get_parameter("uncertainty.extrinsic_stddev_m").value
                    ),
                    world_stddev_m=float(
                        self.get_parameter("uncertainty.world_stddev_m").value
                    ),
                )
                range_m = float(np.linalg.norm(camera_point))
                reference_distance_m = float(
                    self.get_parameter("distance_weighting.reference_distance_m").value
                )
                covariance_scale = max(1.0, (range_m / reference_distance_m) ** 2)
                world_covariance *= covariance_scale

                point_cloud_samples = []
                inlier_ys, inlier_xs = np.where(np.ones_like(diagnostics.inlier_mask))
                num_inliers = len(inlier_ys)
                if num_inliers > 0:
                    sample_size = min(2000, num_inliers)
                    sample_indices = np.random.choice(num_inliers, sample_size, replace=False)
                    for idx in sample_indices:
                        roi_y = inlier_ys[idx]
                        roi_x = inlier_xs[idx]
                        left = diagnostics.bounds_xyxy[0]
                        top = diagnostics.bounds_xyxy[1]
                        u = int(roi_x + left)
                        v = int(roi_y + top)
                        z = float(depth[v, u])
                        if z > 0 and abs(z - depth_m) < adaptive_depth_threshold:
                            pt_cam = deproject_pixel((u, v), z, self._camera_calibration.camera_matrix)
                            pt_world = transform_point(pt_cam, transform.transform)
                            b, g, r = color[v, u]
                            point_cloud_samples.append({
                                "x": float(pt_world[0]),
                                "y": float(pt_world[1]),
                                "z": float(pt_world[2]),
                                "r": int(r),
                                "g": int(g),
                                "b": int(b)
                            })
            except ValueError as exc:
                rejected_depth.append(
                    RejectedDepthDetection(
                        detection.class_name, detection.confidence, detection.bbox_xyxy, str(exc)
                    )
                )
                self.get_logger().debug(
                    f"Rejected {detection.class_name} detection: {exc}"
                )
                continue
            # 3DGS Data Extraction
            left, top, right, bottom = map(int, detection.bbox_xyxy)
            left = max(0, left)
            top = max(0, top)
            right = min(color.shape[1], right)
            bottom = min(color.shape[0], bottom)
            
            if right > left and bottom > top:
                color_crop = color[top:bottom, left:right].copy()
                depth_crop = depth[top:bottom, left:right].copy()
                splat_point_cloud_samples: list[dict[str, float | int]] = []
                
                # [OPTIONAL] ── Gradient-based Depth Segmentation ──────────
                # Uncomment the following block to mask out the background in saved images
                #
                # # 1. Compute depth gradients (depth change per pixel)
                # sobelx = cv2.Sobel(depth_crop, cv2.CV_64F, 1, 0, ksize=3)
                # sobely = cv2.Sobel(depth_crop, cv2.CV_64F, 0, 1, ksize=3)
                # grad_mag = np.sqrt(sobelx**2 + sobely**2)
                # 
                # # 2. Coarse mask: within 0.3m of the estimated median depth
                # coarse_mask = (
                #     np.isfinite(depth_crop)
                #     & (depth_crop > 0)
                #     & (np.abs(depth_crop - depth_m) < 0.3)
                # )
                # 
                # # 3. Disconnect object from background by removing high-gradient edges (> 2cm jump)
                # edge_mask = grad_mag > 0.02
                # core_mask = coarse_mask & ~edge_mask
                # 
                # # 4. Find the object blob using Connected Components
                # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                #     core_mask.astype(np.uint8), connectivity=8
                # )
                # 
                # depth_mask = np.zeros_like(core_mask, dtype=bool)
                # if num_labels > 1:
                #     # Find the blob closest to the center of the bounding box
                #     ch, cw = depth_crop.shape[:2]
                #     center = np.array([cw / 2, ch / 2])
                #     
                #     best_label = 1
                #     min_dist = float('inf')
                #     for i in range(1, num_labels):
                #         # Filter out tiny specks (< 50 pixels)
                #         if stats[i, cv2.CC_STAT_AREA] < 50:
                #             continue
                #         centroid = centroids[i]
                #         dist = np.linalg.norm(centroid - center)
                #         if dist < min_dist:
                #             min_dist = dist
                #             best_label = i
                #             
                #     object_mask = (labels == best_label)
                #     
                #     # 5. Dilate the object mask slightly to recover the edge pixels we removed
                #     morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                #     object_mask_u8 = cv2.dilate(object_mask.astype(np.uint8), morph_kernel, iterations=1)
                #     
                #     # Intersect dilated mask with the original coarse mask
                #     depth_mask = object_mask_u8.astype(bool) & coarse_mask
                #
                # color_crop[~depth_mask] = 0
                # depth_crop[~depth_mask] = 0.0

                
                tx = transform.transform.translation.x
                ty = transform.transform.translation.y
                tz = transform.transform.translation.z
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                
                # Manual quaternion to rotation matrix conversion to avoid tf_transformations dependency
                xx, yy, zz = qx*qx, qy*qy, qz*qz
                xy, xz, yz = qx*qy, qx*qz, qy*qz
                xw, yw, zw = qx*qw, qy*qw, qz*qw
                
                r11 = 1.0 - 2.0 * (yy + zz)
                r12 = 2.0 * (xy - zw)
                r13 = 2.0 * (xz + yw)
                
                r21 = 2.0 * (xy + zw)
                r22 = 1.0 - 2.0 * (xx + zz)
                r23 = 2.0 * (yz - xw)
                
                r31 = 2.0 * (xz - yw)
                r32 = 2.0 * (yz + xw)
                r33 = 1.0 - 2.0 * (xx + yy)
                
                camera_to_world = np.array([
                    [r11, r12, r13, tx],
                    [r21, r22, r23, ty],
                    [r31, r32, r33, tz],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                crop_offset = (left, top)
                splat_point_cloud_samples = self._full_crop_splat_points(
                    color,
                    depth,
                    (left, top, right, bottom),
                    camera_to_world,
                )
            else:
                color_crop = None
                depth_crop = None
                camera_to_world = None
                crop_offset = None
                splat_point_cloud_samples = []

            localized.append(
                LocalizedDetection(
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    class_likelihoods=detection.class_likelihoods,
                    class_evidence_strength=detection.class_evidence_strength,
                    position=tuple(float(value) for value in world_point),
                    position_covariance=world_covariance,
                    stamp_ns=registry_stamp_ns,
                    frame_id=self._reference_frame(),
                    bbox_xyxy=detection.bbox_xyxy,
                    range_m=range_m,
                    appearance_embedding=detection.appearance_embedding,
                    appearance_provider_id=detection.appearance_provider_id,
                    depth_median_m=depth_m,
                    depth_stddev_m=depth_stddev,
                    depth_valid_pixel_count=int(diagnostics.valid_mask.sum()),
                    depth_inlier_pixel_count=int(diagnostics.inlier_mask.sum()),
                    base_pixel_stddev_px=base_pixel_stddev_px,
                    bbox_pixel_stddev_px=bbox_pixel_stddev_px,
                    combined_pixel_stddev_px=combined_pixel_stddev_px,
                    splat_point_cloud=splat_point_cloud_samples,
                    point_cloud=point_cloud_samples,
                    color_crop=color_crop,
                    depth_crop=depth_crop,
                    camera_to_world=camera_to_world,
                    crop_offset=crop_offset,
                )
            )
        timing["depth"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        if bool(self.get_parameter("debug.publish_annotated").value):
            self._publish_annotated_detections(
                color,
                raw_detections,
                packet.color,
                depth_diagnostics,
                rejected_image_detections,
            )
        timing["debug_image"] = time.perf_counter() - stage_started
        self.get_logger().debug(
            "YOLO=%d, localized=%d, depth-rejected=%d, frame=%s, stamp=%d"
            % (
                len(raw_detections),
                len(localized),
                len(rejected_depth),
                source_frame,
                registry_stamp_ns,
            )
        )
        return localized, rejected_depth, tf_mode, Time.from_msg(transform.header.stamp), timing

    def _publish_association_diagnostics(
        self,
        source_image: Image,
        detections: list[LocalizedDetection],
        results: list[AssociationResult],
        rejected: list[RejectedDepthDetection],
        tf_mode: str,
        selected_tf_stamp: Time,
    ) -> None:
        message = AssociationDiagnosticArray()
        message.header = source_image.header
        message.header.frame_id = self._reference_frame()
        requested_stamp = Time.from_msg(source_image.header.stamp)
        selected_stamp_msg = selected_tf_stamp.to_msg()
        stamp_offset_sec = (
            selected_tf_stamp.nanoseconds - requested_stamp.nanoseconds
        ) / 1e9
        for detection, result in zip(detections, results):
            item = AssociationDiagnostic()
            item.header = message.header
            item.class_name = detection.class_name
            item.detector_confidence = float(detection.confidence)
            item.bbox_xyxy = [float(value) for value in detection.bbox_xyxy]
            item.range_m = float(detection.range_m)
            item.depth_median_m = float(detection.depth_median_m)
            item.depth_stddev_m = float(detection.depth_stddev_m)
            item.depth_valid_pixel_count = int(detection.depth_valid_pixel_count)
            item.depth_inlier_pixel_count = int(detection.depth_inlier_pixel_count)
            item.base_pixel_stddev_px = float(detection.base_pixel_stddev_px)
            item.bbox_pixel_stddev_px = float(detection.bbox_pixel_stddev_px)
            item.combined_pixel_stddev_px = float(detection.combined_pixel_stddev_px)
            item.position_covariance_diagonal = [
                float(value) for value in np.diag(detection.position_covariance)
            ]
            item.tf_mode = tf_mode
            item.requested_tf_stamp = source_image.header.stamp
            item.selected_tf_stamp = selected_stamp_msg
            item.tf_stamp_offset_sec = stamp_offset_sec
            item.eligible_track_count = int(result.eligible_track_count)
            item.best_mahalanobis_d2 = _diagnostic_value(result.best_mahalanobis_d2)
            item.selected_mahalanobis_d2 = _diagnostic_value(
                result.selected_mahalanobis_d2
            )
            item.selected_cost = _diagnostic_value(result.selected_cost)
            item.association_decision = result.decision
            item.track_id = result.track.object_id
            message.diagnostics.append(item)
        for rejection in rejected:
            item = AssociationDiagnostic()
            item.header = message.header
            item.class_name = rejection.class_name
            item.detector_confidence = float(rejection.confidence)
            item.bbox_xyxy = [float(value) for value in rejection.bbox_xyxy]
            item.tf_mode = tf_mode
            item.requested_tf_stamp = source_image.header.stamp
            item.selected_tf_stamp = selected_stamp_msg
            item.tf_stamp_offset_sec = stamp_offset_sec
            item.best_mahalanobis_d2 = float("nan")
            item.selected_mahalanobis_d2 = float("nan")
            item.selected_cost = float("nan")
            item.association_decision = "depth_rejected"
            item.rejection_reason = rejection.reason
            message.diagnostics.append(item)
        self._association_diagnostics_publisher.publish(message)

    def _filter_image_detections(
        self, detections: list, image_shape: tuple[int, int]
    ) -> tuple[list, list[RejectedImageDetection]]:
        """Reject implausibly large boxes and boxes clipped by camera edges."""
        image_height, image_width = image_shape
        max_fraction = float(self.get_parameter("detector.max_bbox_area_fraction").value)
        margin_fraction = float(
            self.get_parameter("detector.border_margin_fraction").value
        )
        margin_x = image_width * margin_fraction
        margin_y = image_height * margin_fraction
        accepted = []
        rejected = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            width = max(0.0, min(float(image_width), x2) - max(0.0, x1))
            height = max(0.0, min(float(image_height), y2) - max(0.0, y1))
            fraction = width * height / float(image_width * image_height)
            if fraction > max_fraction:
                reason = f"area={fraction:.1%}>limit={max_fraction:.1%}"
                self.get_logger().debug(
                    f"Rejected {detection.class_name} detection: {reason}"
                )
                rejected.append(
                    RejectedImageDetection(
                        detection.class_name, detection.confidence, detection.bbox_xyxy, reason
                    )
                )
                continue
            if x1 < margin_x or y1 < margin_y or x2 > image_width - margin_x or y2 > image_height - margin_y:
                reason = f"touches_border_margin={margin_fraction:.1%}"
                self.get_logger().debug(
                    f"Rejected {detection.class_name} detection: {reason}"
                )
                rejected.append(
                    RejectedImageDetection(
                        detection.class_name, detection.confidence, detection.bbox_xyxy, reason
                    )
                )
                continue
            accepted.append(detection)
        return accepted, rejected

    def _publish_annotated_detections(
        self,
        color: np.ndarray,
        detections: list,
        source_image: Image,
        depth_diagnostics: dict[int, DepthSamplingDiagnostics],
        rejected_image_detections: list[RejectedImageDetection],
    ) -> None:
        """Publish raw detector boxes from this node's exact inference frame."""
        annotated = color.copy()
        height, width = annotated.shape[:2]
        margin_fraction = float(
            self.get_parameter("detector.border_margin_fraction").value
        )
        margin_x = round(width * margin_fraction)
        margin_y = round(height * margin_fraction)
        cv2.rectangle(
            annotated,
            (margin_x, margin_y),
            (width - margin_x, height - margin_y),
            (255, 0, 255),
            1,
        )
        for index, detection in enumerate(detections):
            left, top, right, bottom = (round(value) for value in detection.bbox_xyxy)
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{detection.class_name} {detection.confidence:.2f}",
                (left, max(20, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            diagnostics = depth_diagnostics.get(index)
            if diagnostics is not None:
                left, top, right, bottom = diagnostics.bounds_xyxy
                cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 255), 1)
                ys, xs = np.where(diagnostics.valid_mask)
                for y, x in zip(ys, xs):
                    color_value = (
                        (255, 255, 0)
                        if diagnostics.inlier_mask[y, x]
                        else (0, 0, 255)
                    )
                    cv2.circle(annotated, (left + int(x), top + int(y)), 1, color_value, -1)
                cv2.drawMarker(
                    annotated,
                    diagnostics.centre_pixel,
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    9,
                    1,
                )
        for rejection in rejected_image_detections:
            left, top, right, bottom = (round(value) for value in rejection.bbox_xyxy)
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"REJECTED:{rejection.reason}",
                (left, max(20, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"Semantic YOLO: {len(detections)} detection(s)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        message = Image()
        message.header = source_image.header
        message.height, message.width = annotated.shape[:2]
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = int(message.width * 3)
        message.data = annotated.tobytes()
        self._debug_image_publisher.publish(message)

    def _publish_observations(
        self, detections: list[LocalizedDetection], source_image: Image
    ) -> None:
        message = DetectedObjectArray()
        message.header.stamp = source_image.header.stamp
        message.header.frame_id = self._reference_frame()
        for detection in detections:
            item = DetectedObject()
            item.header = message.header
            item.class_name = detection.class_name
            item.position.x, item.position.y, item.position.z = detection.position
            item.confidence = detection.confidence
            item.position_stddev_m = float(
                np.sqrt(np.max(np.diag(detection.position_covariance)))
            )
            item.position_covariance = [
                float(value) for value in detection.position_covariance.reshape(-1)
            ]
            item.class_distribution = _distribution_messages(
                detection.class_likelihoods
            )
            item.observation_count = 1
            item.first_seen = source_image.header.stamp
            item.last_seen = source_image.header.stamp
            item.state = "observation"
            message.objects.append(item)
        self._detections_publisher.publish(message)

    def _publish_registry(self) -> None:
        message = DetectedObjectArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._reference_frame()
        with self._registry_lock:
            message.objects = [
                self._track_to_message(track)
                for track in self._registry.tracks(self.get_clock().now().nanoseconds / 1e9)
            ]
        self._objects_publisher.publish(message)

    def _track_to_message(self, track: ObjectTrack) -> DetectedObject:
        message = DetectedObject()
        message.header.frame_id = self._reference_frame()
        message.header.stamp = self.get_clock().now().to_msg()
        message.object_id = track.object_id
        message.class_name = track.class_name
        message.position.x = float(track.position[0])
        message.position.y = float(track.position[1])
        message.position.z = float(track.position[2])
        message.confidence = float(track.confidence)
        message.position_stddev_m = float(track.position_stddev_m)
        message.position_covariance = [
            float(value) for value in track.position_covariance.reshape(-1)
        ]
        message.class_distribution = _distribution_messages(
            track.class_distribution
        )
        message.observation_count = track.observation_count
        message.first_seen = Time(nanoseconds=int(track.first_seen_sec * 1e9)).to_msg()
        message.last_seen = Time(nanoseconds=int(track.last_seen_sec * 1e9)).to_msg()
        message.state = track.state
        return message

    def _search_objects_callback(
        self, request: SearchSemanticObjects.Request, response: SearchSemanticObjects.Response
    ) -> SearchSemanticObjects.Response:
        """Rank confirmed registry tracks against a MobileCLIP text description."""
        response.appearance_provider_id = self._appearance_provider.provider_id
        if not bool(self.get_parameter("semantic_text.enabled").value):
            response.success = False
            response.reason = "Semantic text search is disabled"
            return response
        query = str(request.query).strip()
        if not query:
            response.success = False
            response.reason = "Text query must not be empty"
            return response
        configured_limit = int(self.get_parameter("semantic_text.max_results").value)
        limit = configured_limit if request.max_results == 0 else min(
            int(request.max_results), configured_limit
        )
        configured_threshold = float(
            self.get_parameter("semantic_text.minimum_cosine_similarity").value
        )
        threshold = (
            configured_threshold
            if float(request.minimum_cosine_similarity) < 0.0
            else float(request.minimum_cosine_similarity)
        )
        response.applied_minimum_cosine_similarity = threshold
        if not -1.0 <= threshold <= 1.0:
            response.success = False
            response.reason = "minimum_cosine_similarity must be in [-1, 1]"
            return response
        try:
            text_embedding = self._appearance_provider.encode_text(query)
        except RuntimeError as exc:
            response.success = False
            response.reason = str(exc)
            return response
        text_embedding = np.asarray(text_embedding, dtype=np.float64).reshape(-1)
        text_norm = float(np.linalg.norm(text_embedding))
        if (
            text_embedding.size == 0
            or not np.all(np.isfinite(text_embedding))
            or text_norm <= 0.0
        ):
            response.success = False
            response.reason = "Appearance provider returned an invalid text embedding"
            return response
        text_embedding /= text_norm

        compatible: list[tuple[float, ObjectTrack]] = []
        provider_id = self._appearance_provider.provider_id
        with self._registry_lock:
            for track in self._registry.tracks():
                embedding = track.appearance_embedding
                if (
                    track.state != "confirmed"
                    or embedding is None
                    or track.appearance_provider_id != provider_id
                ):
                    continue
                vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
                norm = float(np.linalg.norm(vector))
                if (
                    vector.shape != text_embedding.shape
                    or not np.all(np.isfinite(vector))
                    or norm <= 0.0
                ):
                    continue
                similarity = float(np.dot(text_embedding, vector / norm))
                if np.isfinite(similarity):
                    compatible.append((float(np.clip(similarity, -1.0, 1.0)), track))
        response.compatible_track_count = len(compatible)
        compatible.sort(
            key=lambda item: (
                -item[0],
                -float(item[1].confidence),
                -float(item[1].last_seen_sec),
                item[1].object_id,
            )
        )
        for similarity, track in compatible:
            if similarity < threshold:
                continue
            match = SemanticTextMatch()
            match.object_id = track.object_id
            match.class_name = track.class_name
            match.cosine_similarity = similarity
            match.confidence = float(track.confidence)
            match.position.x = float(track.position[0])
            match.position.y = float(track.position[1])
            match.position.z = float(track.position[2])
            match.position_stddev_m = float(track.position_stddev_m)
            match.last_seen = Time(nanoseconds=int(track.last_seen_sec * 1e9)).to_msg()
            match.state = track.state
            response.matches.append(match)
            if len(response.matches) >= limit:
                break
        response.success = True
        response.reason = ""
        return response

    def _find_object_goal_callback(self, request: FindObject.Goal) -> GoalResponse:
        if not request.target_class.strip() or request.min_confidence > 1.0:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_find_object(self, goal_handle: object) -> FindObject.Result:
        target = goal_handle.request.target_class.strip().lower().replace(" ", "_")
        min_confidence = float(goal_handle.request.min_confidence)
        if min_confidence <= 0.0:
            min_confidence = float(
                self.get_parameter("detector.confidence_threshold").value
            )
        timeout_sec = float(goal_handle.request.timeout_sec)
        if timeout_sec <= 0.0:
            timeout_sec = float(self.get_parameter("search.default_timeout_sec").value)
        start_ros_sec = self.get_clock().now().nanoseconds / 1e9
        deadline = time.monotonic() + timeout_sec
        last_feedback_stamp = -1.0
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._find_result(False, None, "Search was cancelled")
            with self._registry_changed:
                match = self._registry.best_confirmed(
                    target, min_confidence, seen_since_sec=start_ros_sec
                )
                if match is not None:
                    self._target_found_publisher.publish(self._track_to_message(match))
                    goal_handle.succeed()
                    return self._find_result(True, match, "Confirmed target detected")
                candidate = self._registry.best_candidate(target)
                if (
                    candidate is not None
                    and candidate.last_seen_sec >= start_ros_sec
                    and candidate.last_seen_sec > last_feedback_stamp
                ):
                    feedback = FindObject.Feedback()
                    feedback.candidate = self._track_to_message(candidate)
                    goal_handle.publish_feedback(feedback)
                    last_feedback_stamp = candidate.last_seen_sec
                remaining = max(0.0, deadline - time.monotonic())
                self._registry_changed.wait(timeout=min(0.1, remaining))
        goal_handle.abort()
        return self._find_result(False, None, "Timed out without a confirmed target")

    def _find_result(
        self, found: bool, track: ObjectTrack | None, reason: str
    ) -> FindObject.Result:
        result = FindObject.Result()
        result.found = found
        result.reason = reason
        if track is not None:
            result.object = self._track_to_message(track)
        return result

    def _load_registry(self) -> None:
        try:
            if self._registry_path.is_file():
                if self._registry_path.stat().st_size == 0:
                    self.get_logger().warning(
                        f"Semantic registry is empty; starting with an empty registry: "
                        f"{self._registry_path}"
                    )
                    return
                self._registry.load(self._registry_path)
            elif self._legacy_path.is_file():
                self._registry.load(self._legacy_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not load semantic registry: {exc}") from exc

    def _persist_if_due(self) -> None:
        with self._registry_lock:
            if not self._registry_dirty:
                return
            interval = float(
                self.get_parameter("registry.persistence_interval_sec").value
            )
            if time.monotonic() - self._last_persist_monotonic < interval:
                return
            try:
                self._registry.persist(self._registry_path, self._reference_frame())
                if self.get_parameter("registry.write_legacy_coordinates").value:
                    self._registry.persist_legacy_coordinates(self._legacy_path)
            except OSError as exc:
                self.get_logger().error(f"Could not persist semantic registry: {exc}")
                return
            self._registry_dirty = False
            self._last_persist_monotonic = time.monotonic()

    def _reference_frame(self) -> str:
        return str(self.get_parameter("reference_frame").value).strip()

    def _camera_calibration_from_parameters(self) -> FixedCameraCalibration:
        return FixedCameraCalibration(
            fx=float(self.get_parameter("camera.fx").value),
            fy=float(self.get_parameter("camera.fy").value),
            cx=float(self.get_parameter("camera.cx").value),
            cy=float(self.get_parameter("camera.cy").value),
            image_width=int(self.get_parameter("camera.image_width").value),
            image_height=int(self.get_parameter("camera.image_height").value),
        )

    def destroy_node(self) -> bool:
        with self._registry_lock:
            if self._registry_dirty:
                try:
                    self._registry.persist(self._registry_path, self._reference_frame())
                    if self.get_parameter("registry.write_legacy_coordinates").value:
                        self._registry.persist_legacy_coordinates(self._legacy_path)
                except OSError as exc:
                    self.get_logger().error(f"Final registry write failed: {exc}")
        self._action_server.destroy()
        self.destroy_service(self._text_search_service)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: SemanticPerceptionNode | None = None
    try:
        node = SemanticPerceptionNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _distribution_messages(distribution: dict[str, float]) -> list[ClassProbability]:
    total = sum(max(0.0, float(value)) for value in distribution.values())
    if total <= 0.0:
        return []
    messages: list[ClassProbability] = []
    for label, value in sorted(distribution.items()):
        item = ClassProbability()
        item.class_name = str(label)
        item.probability = max(0.0, float(value)) / total
        messages.append(item)
    return messages


def _diagnostic_value(value: float | None) -> float:
    return float(value) if value is not None else float("nan")


def _decode_color_image(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    channels_by_encoding = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported RGB encoding: {message.encoding}")
    required_row_bytes = int(message.width) * channels
    if message.step < required_row_bytes:
        raise ValueError("RGB image step is smaller than its encoded width")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
        int(message.height), int(message.step)
    )
    image = rows[:, :required_row_bytes].reshape(
        int(message.height), int(message.width), channels
    )
    if encoding == "rgb8" or encoding == "rgba8":
        image = image[:, :, [2, 1, 0]]
    elif encoding == "bgra8":
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def _decode_depth_image(message: Image) -> np.ndarray:
    encoding = message.encoding.upper()
    dtype_by_encoding = {
        "16UC1": np.dtype(np.uint16),
        "MONO16": np.dtype(np.uint16),
        "32FC1": np.dtype(np.float32),
        "64FC1": np.dtype(np.float64),
    }
    base_dtype = dtype_by_encoding.get(encoding)
    if base_dtype is None:
        raise ValueError(f"Unsupported depth encoding: {message.encoding}")
    dtype = base_dtype.newbyteorder(">" if message.is_bigendian else "<")
    required_row_bytes = int(message.width) * dtype.itemsize
    if message.step < required_row_bytes:
        raise ValueError("Depth image step is smaller than its encoded width")
    byte_rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
        int(message.height), int(message.step)
    )
    compact = np.ascontiguousarray(byte_rows[:, :required_row_bytes])
    depth = compact.view(dtype).reshape(int(message.height), int(message.width))
    return depth.astype(np.float32, copy=False)


if __name__ == "__main__":
    main()
