"""ROS 2 RGB-D semantic perception node."""

from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np

import message_filters
import rclpy
from interface.action import FindObject
from interface.msg import ClassProbability, DetectedObject, DetectedObjectArray
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
from .localization import (
    LocalizedDetection,
    FixedCameraCalibration,
    deproject_pixel,
    deprojection_covariance,
    robust_depth_at_detection,
    transform_covariance,
    transform_point,
)
from .registry import ObjectRegistry, ObjectTrack, RegistryConfig


@dataclass(frozen=True)
class SensorPacket:
    sequence: int
    color: Image
    depth: Image


class SemanticPerceptionNode(Node):
    def __init__(self, detector: ObjectDetector | None = None) -> None:
        super().__init__("semantic_perception")
        self._declare_parameters()
        self._validate_parameters()
        self._packet_lock = threading.Lock()
        self._registry_lock = threading.RLock()
        self._registry_changed = threading.Condition(self._registry_lock)
        self._latest_packet: SensorPacket | None = None
        self._received_sequence = 0
        self._processed_sequence = 0
        self._color_messages_received = 0
        self._depth_messages_received = 0
        self._synchronized_packets_received = 0
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
                evidence_decay=float(self.get_parameter("class.evidence_decay").value),
                process_noise_stddev_m=float(
                    self.get_parameter("uncertainty.process_noise_stddev_m").value
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
        self._input_health_timer = self.create_timer(1.0, self._log_input_health)
        self._action_server = ActionServer(
            self,
            FindObject,
            "/semantic/find_object",
            execute_callback=self._execute_find_object,
            goal_callback=self._find_object_goal_callback,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self._publish_registry()
        self.get_logger().info(
            f"Semantic perception ready: model={self._detector.model_id}, "
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
        self.declare_parameter("sync.slop_sec", 0.05)
        self.declare_parameter("tf_timeout_sec", 0.2)
        self.declare_parameter("tf.fallback_to_latest", True)
        self.declare_parameter("detector.backend", "ultralytics")
        self.declare_parameter("detector.model_path", "")
        self.declare_parameter("detector.device", "cuda:0")
        self.declare_parameter("detector.image_size", 640)
        self.declare_parameter("detector.confidence_threshold", 0.50)
        self.declare_parameter("detector.max_detections", 100)
        self.declare_parameter("detector.max_bbox_area_fraction", 0.55)
        self.declare_parameter("depth.scale_16uc1", 0.001)
        self.declare_parameter("depth.inner_bbox_fraction", 0.5)
        self.declare_parameter("depth.minimum_valid_pixels", 20)
        self.declare_parameter("depth.minimum_m", 0.05)
        self.declare_parameter("depth.maximum_m", 5.0)
        self.declare_parameter("association.mahalanobis_threshold", 11.345)
        self.declare_parameter("filter.confirmation_hits", 3)
        self.declare_parameter("filter.confirmation_window_sec", 3.0)
        self.declare_parameter("filter.stale_after_sec", 300.0)
        self.declare_parameter("class.confirmation_probability", 0.70)
        self.declare_parameter("class.max_explicit_classes", 4)
        self.declare_parameter("class.evidence_decay", 0.95)
        self.declare_parameter("uncertainty.pixel_stddev_px", 2.0)
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
        self.declare_parameter("debug.publish_annotated", False)
        self.declare_parameter("debug.annotated_topic", "/semantic/debug/yolo")

    def _validate_parameters(self) -> None:
        positive_parameters = (
            "processing_rate_hz",
            "sync.queue_size",
            "sync.slop_sec",
            "tf_timeout_sec",
            "detector.image_size",
            "detector.max_detections",
            "depth.minimum_valid_pixels",
            "depth.maximum_m",
            "association.mahalanobis_threshold",
            "filter.confirmation_hits",
            "filter.confirmation_window_sec",
            "filter.stale_after_sec",
            "class.max_explicit_classes",
            "uncertainty.pixel_stddev_px",
            "uncertainty.process_noise_stddev_m",
            "registry.persistence_interval_sec",
            "search.default_timeout_sec",
        )
        for name in positive_parameters:
            if float(self.get_parameter(name).value) <= 0.0:
                raise ValueError(f"ROS parameter {name} must be positive")
        unit_parameters = (
            "detector.confidence_threshold",
            "detector.max_bbox_area_fraction",
            "depth.inner_bbox_fraction",
            "class.confirmation_probability",
            "class.evidence_decay",
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
        with self._packet_lock:
            self._received_sequence += 1
            self._synchronized_packets_received += 1
            self._latest_packet = SensorPacket(
                sequence=self._received_sequence,
                color=color,
                depth=depth,
            )

    def _color_received(self, _: Image) -> None:
        self._color_messages_received += 1

    def _depth_received(self, _: Image) -> None:
        self._depth_messages_received += 1

    def _log_input_health(self) -> None:
        self.get_logger().info(
            "RGB-D input health: color=%d depth=%d synchronized=%d processed=%d"
            % (
                self._color_messages_received,
                self._depth_messages_received,
                self._synchronized_packets_received,
                self._processed_sequence,
            )
        )

    def _process_latest(self) -> None:
        with self._packet_lock:
            packet = self._latest_packet
            if packet is None or packet.sequence == self._processed_sequence:
                return
            self._processed_sequence = packet.sequence
        try:
            localized = self._localize_packet(packet)
        except Exception as exc:
            self.get_logger().warning(f"Dropped RGB-D packet: {exc}")
            return
        self._publish_observations(localized, packet.color)
        changed_tracks: list[ObjectTrack] = []
        with self._registry_changed:
            changed_tracks = self._registry.update_frame(localized)
            if changed_tracks:
                self._registry_dirty = True
                self._registry_changed.notify_all()
        if changed_tracks:
            self._publish_registry()
        self._persist_if_due()

    def _localize_packet(self, packet: SensorPacket) -> list[LocalizedDetection]:
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
        source_frame = str(self.get_parameter("camera.tf_frame_override").value).strip()
        if not source_frame:
            source_frame = packet.color.header.frame_id.strip()
        if not source_frame:
            raise ValueError("RGB image has no frame_id")
        stamp = Time.from_msg(packet.color.header.stamp)
        try:
            transform = self._tf_buffer.lookup_transform(
                self._reference_frame(),
                source_frame,
                stamp,
                timeout=Duration(
                    seconds=float(self.get_parameter("tf_timeout_sec").value)
                ),
            )
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
        stamp_ns = stamp.nanoseconds
        raw_detections = self._filter_large_detections(
            self._detector.detect(color), color.shape[:2]
        )
        if bool(self.get_parameter("debug.publish_annotated").value):
            self._publish_annotated_detections(color, raw_detections, packet.color)
        localized: list[LocalizedDetection] = []
        rejected_depth = 0
        for detection in raw_detections:
            try:
                depth_m, depth_stddev, pixel = robust_depth_at_detection(
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
                camera_point = deproject_pixel(
                    pixel, depth_m, self._camera_calibration.camera_matrix
                )
                camera_covariance = deprojection_covariance(
                    pixel,
                    depth_m,
                    depth_stddev,
                    self._camera_calibration.camera_matrix,
                    pixel_stddev_px=float(
                        self.get_parameter("uncertainty.pixel_stddev_px").value
                    ),
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
            except ValueError as exc:
                rejected_depth += 1
                self.get_logger().debug(
                    f"Rejected {detection.class_name} detection: {exc}"
                )
                continue
            localized.append(
                LocalizedDetection(
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    class_likelihoods=detection.class_likelihoods,
                    position=tuple(float(value) for value in world_point),
                    position_covariance=world_covariance,
                    stamp_ns=stamp_ns,
                    frame_id=self._reference_frame(),
                    bbox_xyxy=detection.bbox_xyxy,
                )
            )
        self.get_logger().debug(
            "YOLO=%d, localized=%d, depth-rejected=%d, frame=%s, stamp=%d"
            % (
                len(raw_detections),
                len(localized),
                rejected_depth,
                source_frame,
                stamp_ns,
            )
        )
        return localized

    def _filter_large_detections(self, detections: list, image_shape: tuple[int, int]) -> list:
        """Reject boxes that cover implausibly large portions of the image."""
        image_height, image_width = image_shape
        max_fraction = float(self.get_parameter("detector.max_bbox_area_fraction").value)
        accepted = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            width = max(0.0, min(float(image_width), x2) - max(0.0, x1))
            height = max(0.0, min(float(image_height), y2) - max(0.0, y1))
            fraction = width * height / float(image_width * image_height)
            if fraction > max_fraction:
                self.get_logger().debug(
                    f"Rejected {detection.class_name} detection: box covers "
                    f"{fraction:.1%} of the image (limit {max_fraction:.1%})"
                )
                continue
            accepted.append(detection)
        return accepted

    def _publish_annotated_detections(
        self, color: np.ndarray, detections: list, source_image: Image
    ) -> None:
        """Publish raw detector boxes from this node's exact inference frame."""
        annotated = color.copy()
        for detection in detections:
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
