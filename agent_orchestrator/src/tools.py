import base64
import binascii
import json
import math
import os
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

from config import (
    YOLOConfig,
    RuntimeConfig,
    RuntimeConfigurationError,
    get_runtime_config,
)
from ros_interface import (
    get_shared_node,
    wait_for_grasp,
    wait_for_joint_target,
    wait_for_place,
)
from targeted_scan_geometry import (
    ScanPose,
    close_view_tilt_poses,
    generate_desk_arc,
    generate_desk_targets,
    generate_object_arc_angles,
    look_at_eef_rotation,
    object_arc_pose,
    rail_centre,
)


VLM_WARMUP_LATENCY_LIMIT_SEC = 5.0
MAX_REACQUISITION_FRAMES = 2
SEARCH_TELEMETRY_TIMEOUT_SEC = 10.0
JOINT_MOTION_TIMEOUT_SEC = 50.0
HOME_MOTION_TIMEOUT_SEC = 120.0
JOINT_POLL_INTERVAL_SEC = 0.05
# Keep a quiet ownership handoff interval between a completed direct full-arm
# posture and the next Cartesian/RRT request.  The bridge reports release
# before all downstream simulator callbacks have necessarily observed it.
DIRECT_ARM_POSTURE_HANDOFF_DELAY_SEC = 0.25
# The rail is positioned from a semantic-map estimate.  Leave enough lateral
# workspace for normal map/rail feedback error while still rejecting targets
# that would require a materially different rail alignment.
MAX_MANIPULATION_RAIL_OFFSET_M = 0.25
STARTUP_RAIL_POSITION_M = -1.1
STARTUP_ARM_JOINTS = (0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854)
_yolo_detector = None


def _normalize_label(label: object) -> str:
    return str(label).strip().lower()


def _decode_ros_image(encoded_image: str) -> np.ndarray:
    if not isinstance(encoded_image, str) or not encoded_image.strip():
        raise RuntimeConfigurationError("ROS camera returned an empty image payload")

    payload = encoded_image.strip()
    if payload.startswith("data:"):
        try:
            payload = payload.split(",", 1)[1]
        except IndexError as exc:
            raise RuntimeConfigurationError("ROS camera returned an invalid data URL") from exc

    try:
        encoded_bytes = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RuntimeConfigurationError("ROS camera image is not valid base64") from exc

    frame = cv2.imdecode(np.frombuffer(encoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.ndim < 2 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise RuntimeConfigurationError("ROS camera image could not be decoded")
    return frame


def _tensor_values(value: object) -> list:
    """Convert an Ultralytics tensor or NumPy value to plain Python values."""
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else list(value)


def _class_label(names: object, class_id: int) -> str:
    if isinstance(names, Mapping):
        label = names.get(class_id, names.get(str(class_id), ""))
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        label = names[class_id] if 0 <= class_id < len(names) else ""
    else:
        label = ""
    return _normalize_label(label)


class DeterministicYOLODetector:
    """Run configured local YOLO inference and enforce the detection contract."""

    def __init__(self, config: YOLOConfig, *, model: object | None = None) -> None:
        self.config = config
        self.model = model if model is not None else self._load_model()

    def _load_model(self) -> object:
        if not self.config.checkpoint_path.is_file():
            raise RuntimeConfigurationError(
                f"YOLO checkpoint does not exist: {self.config.checkpoint_path}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeConfigurationError(
                "The 'ultralytics' package is required for local YOLO detection"
            ) from exc
        try:
            return YOLO(str(self.config.checkpoint_path), task="detect")
        except Exception as exc:
            raise RuntimeConfigurationError(
                f"Could not load YOLO checkpoint {self.config.checkpoint_path}: {exc}"
            ) from exc

    def supports_label(self, target_label: str) -> bool:
        normalized_target = _normalize_label(target_label)
        names = getattr(self.model, "names", {})
        if isinstance(names, Mapping):
            labels = names.values()
        elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            labels = names
        else:
            return False
        return normalized_target in {_normalize_label(label) for label in labels}

    def detect(self, frame: np.ndarray, target_label: str) -> dict[str, object]:
        normalized_target = _normalize_label(target_label)
        if not normalized_target:
            raise RuntimeConfigurationError("YOLO target label must not be empty")
        if frame.ndim < 2 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise RuntimeConfigurationError("YOLO received an invalid camera frame")

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.config.image_size,
                conf=0.0,
                device=self.config.device,
                max_det=self.config.max_detections,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeConfigurationError(f"YOLO inference failed: {exc}") from exc

        image_height, image_width = frame.shape[:2]
        candidates: list[dict[str, object]] = []

        node = get_shared_node()
        current_rail = getattr(node, "current_rail_position", None)
        rail_text = f"Robot Rail: {current_rail:.4f}m" if current_rail is not None else "Robot Rail: N/A"
        cv2.putText(frame, rail_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if results:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                coordinates = _tensor_values(boxes.xyxy)
                confidences = _tensor_values(boxes.conf)
                class_ids = _tensor_values(boxes.cls)
                names = getattr(result, "names", getattr(self.model, "names", {}))

                for coordinates_row, confidence_value, class_value in zip(
                    coordinates, confidences, class_ids
                ):
                    if len(coordinates_row) != 4:
                        continue
                    x1, y1, x2, y2 = map(float, coordinates_row)
                    confidence = float(confidence_value)
                    if confidence <= 0.0:
                        continue
                    values = (x1, y1, x2, y2, confidence)
                    if not all(math.isfinite(value) for value in values):
                        continue

                    try:
                        numeric_class_id = float(class_value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(numeric_class_id):
                        continue
                    label = _class_label(names, int(numeric_class_id))
                    
                    if confidence > 0.3:
                        color = (0, 255, 0) if label == normalized_target else (0, 0, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        text = f"{label} {confidence:.2f}"
                        cv2.putText(frame, text, (int(x1), max(10, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if label != normalized_target:
                        continue
                    if x1 <= 0.0 or y1 <= 0.0 or x2 >= image_width or y2 >= image_height:
                        continue
                    if x2 <= x1 or y2 <= y1:
                        continue

                    area = (x2 - x1) * (y2 - y1)
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    horizontal_error = (center_x - image_width / 2.0) / (
                        image_width / 2.0
                    )
                    final_eligible = confidence > self.config.confidence_threshold
                    candidates.append(
                        {
                            "status": "accepted" if final_eligible else "guidance",
                            "label": label,
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],
                            "area": area,
                            "center": (center_x, center_y),
                            "horizontal_error": horizontal_error,
                            "final_eligible": final_eligible,
                            "guidance_only": not final_eligible,
                        }
                    )
                    
        scale = 0.6
        vis_width = int(frame.shape[1] * scale)
        vis_height = int(frame.shape[0] * scale)
        vis_frame = cv2.resize(frame, (vis_width, vis_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("YOLO Search", vis_frame)
        cv2.waitKey(1)

        if not candidates:
            return {
                "status": "not_found",
                "label": normalized_target,
                "confidence": None,
                "bbox": None,
                "area": None,
                "center": None,
                "horizontal_error": None,
                "final_eligible": False,
                "guidance_only": False,
            }
        return max(
            candidates,
            key=lambda candidate: (
                float(candidate["confidence"]),
                float(candidate["area"]),
            ),
        )


def _get_yolo_detector(config: RuntimeConfig) -> DeterministicYOLODetector:
    global _yolo_detector
    if _yolo_detector is None or _yolo_detector.config != config.yolo:
        _yolo_detector = DeterministicYOLODetector(config.yolo)
    return _yolo_detector


def detect_target_in_latest_frame(
    target_label: str, *, timeout_sec: float = 10.0
) -> dict[str, object]:
    """Detect a target in one stationary wrist-camera observation.

    The Step 5 search state machine is responsible for ensuring the robot is
    stationary before calling this internal observation function.
    """
    config = get_runtime_config()
    encoded_image = get_latest_ros_image(timeout_sec=timeout_sec)
    frame = _decode_ros_image(encoded_image)
    return _get_yolo_detector(config).detect(frame, target_label)


def is_vlm_server_responsive(config: RuntimeConfig) -> bool:
    """Return True only when llama.cpp's HTTP health endpoint is ready."""
    try:
        with urllib.request.urlopen(
            config.vlm.health_url, timeout=config.vlm.health_timeout_sec
        ) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _get_vlm_server_model_ids(config: RuntimeConfig) -> set[str]:
    try:
        with urllib.request.urlopen(
            f"{config.vlm.api_base_url}/models",
            timeout=config.vlm.health_timeout_sec,
        ) as response:
            payload = json.load(response)
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        raise RuntimeConfigurationError(
            f"Could not query VLM model metadata at {config.vlm.api_base_url}/models: {exc}"
        ) from exc

    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeConfigurationError(
            "VLM /v1/models response did not contain a 'data' list"
        )
    return {
        str(model.get("id"))
        for model in models
        if isinstance(model, dict) and model.get("id") is not None
    }


def _run_vlm_warmup(config: RuntimeConfig) -> float:
    request_body = json.dumps(
        {
            "model": config.vlm.model_alias,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{config.vlm.api_base_url}/chat/completions",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(
            request, timeout=VLM_WARMUP_LATENCY_LIMIT_SEC
        ) as response:
            response_payload = json.load(response)
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        raise RuntimeConfigurationError(
            "VLM warm-up request failed before the 5-second latency gate: "
            f"{exc}"
        ) from exc

    elapsed_sec = time.perf_counter() - started_at
    choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeConfigurationError(
            "VLM warm-up response did not contain a non-empty 'choices' list"
        )
    if elapsed_sec >= VLM_WARMUP_LATENCY_LIMIT_SEC:
        raise RuntimeConfigurationError(
            "VLM warm-up latency must be strictly below 5 seconds, got "
            f"{elapsed_sec:.3f} seconds"
        )
    return elapsed_sec


def validate_server_readiness(config: RuntimeConfig) -> None:
    if not is_vlm_server_responsive(config):
        raise RuntimeConfigurationError(
            "VLM server is not healthy at "
            f"{config.vlm.health_url}; no robot motion will be started"
        )
    model_ids = _get_vlm_server_model_ids(config)
    if config.vlm.model_alias not in model_ids:
        raise RuntimeConfigurationError(
            f"VLM server is responsive but does not expose configured model alias "
            f"{config.vlm.model_alias!r}; available model IDs: {sorted(model_ids)}"
        )
    warmup_latency_sec = _run_vlm_warmup(config)
    print(
        "[SYSTEM] VLM warm-up completed in "
        f"{warmup_latency_sec:.3f}s (<{VLM_WARMUP_LATENCY_LIMIT_SEC:g}s)."
    )


def _wait_for_vlm_server(config: RuntimeConfig) -> bool:
    deadline = time.monotonic() + config.vlm.startup_timeout_sec
    while time.monotonic() < deadline:
        if is_vlm_server_responsive(config):
            return True
        time.sleep(0.5)
    return False


def start_vlm_server(config: RuntimeConfig | None = None) -> bool:
    """Start the configured local llama.cpp server and require HTTP readiness."""
    config = get_runtime_config() if config is None else config
    check = subprocess.run(
        ["tmux", "has-session", "-t", "vlm_server"], capture_output=True
    )
    if check.returncode == 0:
        if is_vlm_server_responsive(config):
            try:
                validate_server_readiness(config)
            except RuntimeConfigurationError:
                pass
            else:
                print(
                    f"[SYSTEM] VLM server is already healthy at {config.vlm.health_url}."
                )
                return True
        print(
            "[SYSTEM] tmux session 'vlm_server' is unhealthy or does not match "
            "the configured model alias; restarting it."
        )
        subprocess.run(
            ["tmux", "kill-session", "-t", "vlm_server"], capture_output=True
        )

    server_command = [
        str(config.vlm.executable),
        "-m",
        str(config.vlm.model_path),
        "--mmproj",
        str(config.vlm.mmproj_path),
        "-ngl",
        str(config.vlm.gpu_layers),
        "-c",
        str(config.vlm.context_size),
        "-np",
        str(config.vlm.parallel_slots),
        "-fa",
        "on" if config.vlm.flash_attention else "off",
        "-n",
        str(config.vlm.max_completion_tokens),
        "--jinja",
        "--host",
        config.vlm.host,
        "--port",
        str(config.vlm.port),
        "--alias",
        config.vlm.model_alias,
        *config.vlm.extra_server_args,
    ]

    print(
        f"[SYSTEM] Launching {config.vlm.model_alias} with a "
        f"{config.vlm.vram_budget_gb:g}GB VLM budget "
        f"({config.vlm.total_gpu_vram_gb:g}GB GPU total)..."
    )
    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", "vlm_server"], check=True
        )
        subprocess.run(
            [
                "tmux",
                "send-keys",
                "-t",
                "vlm_server",
                shlex.join(server_command),
                "C-m",
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeConfigurationError(f"Could not launch VLM server: {exc}") from exc

    if not _wait_for_vlm_server(config):
        raise RuntimeConfigurationError(
            f"VLM server did not become healthy at {config.vlm.health_url} within "
            f"{config.vlm.startup_timeout_sec:g} seconds"
        )

    validate_server_readiness(config)
    print(f"[SYSTEM] VLM server is ready at {config.vlm.health_url}.")
    return True


def stop_vlm_server():
    """Stops the llama.cpp VLM server tmux session."""
    print("[SYSTEM] Terminating 'vlm_server' tmux session...")
    subprocess.run(['tmux', 'kill-session', '-t', 'vlm_server'], capture_output=True)



# ── TOOL EXECUTION WRAPPERS ──

def start_joint_controller() -> str:
    """Initialize ROS and preserve a previously confirmed held object."""
    try:
        config = get_runtime_config()
        node = get_shared_node()
        if not _wait_for_startup_pose_telemetry(node):
            return "Error starting controller: timed out waiting for rail and arm joint telemetry."
        # Startup must take over cleanly even when the previous task stopped
        # after an RRT command and this process has no matching local status.
        if not node.cancel_eef_motion(config.search.targeted_cancel_timeout_sec):
            print(
                "[STARTUP] No RRT cancellation acknowledgement; continuing with "
                "the direct safe-posture takeover."
            )
        node.cartesian_control_active = False

        _command_default_standing_posture_and_wait(node, config)

        node.cartesian_control_active = False
        gripper_feedback_confirmed = True
        if _held_object_id is None:
            try:
                _command_gripper_and_wait(
                    node,
                    config.manipulation.gripper_open_command,
                    config.manipulation.gripper_open_state_m,
                    config,
                )
            except RuntimeConfigurationError as exc:
                # The command has already been published.  Startup remains usable
                # when gripper telemetry starts late; pick/place retain strict
                # feedback checks before performing a manipulation.
                gripper_feedback_confirmed = False
                print(f"[STARTUP] Open-gripper feedback not confirmed: {exc}")
        return (
            "Success: controller is ready at startup pose "
            f"(rail_j1 remains at {float(node.current_rail_position):.4f}m) "
            + (
                f"while holding {_held_object_id}; gripper remains closed"
                if _held_object_id is not None
                else "with open-gripper command issued"
            )
            + ("." if gripper_feedback_confirmed else "; feedback is pending.")
        )
    except Exception as e:
        return f"Error starting controller: {e}"


def _wait_for_startup_pose_telemetry(node: object) -> bool:
    """Require feedback for the rail and every arm joint before startup motion."""
    required_joints = {"rail_j1", *(f"panda_joint{i}" for i in range(1, 8))}
    deadline = time.monotonic() + SEARCH_TELEMETRY_TIMEOUT_SEC
    while time.monotonic() < deadline:
        positions = getattr(node, "current_joint_positions", {})
        if required_joints.issubset(positions) and all(
            math.isfinite(float(positions[joint])) for joint in required_joints
        ):
            return True
        time.sleep(JOINT_POLL_INTERVAL_SEC)
    return False


def _failure_result(
    target: str,
    reason: str,
    *,
    state: str,
    detection: Mapping[str, object] | None = None,
    observations: int = 0,
    recoverable: bool = False,
) -> dict[str, object]:
    detection = detection or {}
    return {
        "status": "failure",
        "success": False,
        "state": state,
        "reason": reason,
        "target": target,
        "confidence": detection.get("confidence"),
        "bbox": detection.get("bbox"),
        "center": detection.get("center"),
        "horizontal_error": detection.get("horizontal_error"),
        "x": None,
        "observations": observations,
        "recoverable": recoverable,
    }


def _wait_for_search_telemetry(node: object) -> bool:
    deadline = time.monotonic() + SEARCH_TELEMETRY_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if (
            getattr(node, "current_rail_position", None) is not None
            and getattr(node, "current_panda_joint1", None) is not None
        ):
            return True
        time.sleep(0.05)
    return False


def _segment_waypoints(start: float, end: float, spacing: float) -> list[float]:
    if math.isclose(start, end, abs_tol=1e-12):
        return []
    direction = 1.0 if end > start else -1.0
    position = start
    waypoints: list[float] = []
    while abs(end - position) > spacing + 1e-12:
        position += direction * spacing
        waypoints.append(position)
    if not waypoints or not math.isclose(waypoints[-1], end, abs_tol=1e-12):
        waypoints.append(end)
    return waypoints


def _build_rail_search_waypoints(
    current_position: float, config: RuntimeConfig
) -> list[float]:
    search = config.search
    if not search.rail_min_position <= current_position <= search.rail_max_position:
        raise RuntimeConfigurationError(
            f"Current rail position {current_position:.4f}m is outside configured "
            f"bounds [{search.rail_min_position:.4f}, {search.rail_max_position:.4f}]m"
        )

    distance_to_min = current_position - search.rail_min_position
    distance_to_max = search.rail_max_position - current_position
    if distance_to_min <= distance_to_max:
        first_limit = search.rail_min_position
        opposite_limit = search.rail_max_position
    else:
        first_limit = search.rail_max_position
        opposite_limit = search.rail_min_position

    return [
        first_limit,
        *_segment_waypoints(
            first_limit, opposite_limit, search.rail_waypoint_spacing
        ),
    ]


def _command_rail_and_wait(
    node: object,
    target: float,
    config: RuntimeConfig,
    *,
    timeout_sec: float = JOINT_MOTION_TIMEOUT_SEC,
) -> None:
    if not config.search.rail_min_position <= target <= config.search.rail_max_position:
        raise RuntimeConfigurationError(
            f"Refusing rail target {target:.4f}m outside configured bounds"
        )
    ensure_rrt_idle = getattr(node, "ensure_rrt_idle", None)
    if callable(ensure_rrt_idle) and not ensure_rrt_idle(
        config.search.targeted_cancel_timeout_sec
    ):
        raise RuntimeConfigurationError(
            "Could not stop active RRT trajectory before rail motion"
        )
    starting_position = getattr(node, "current_rail_position", None)
    current_position = (
        float(starting_position) if starting_position is not None else target
    )
    direction = (
        1.0 if target > current_position else -1.0 if target < current_position else 0.0
    )
    node.send_absolute_rail_command(
        float(target), speed=config.search.rail_speed * direction
    )
    if not _wait_for_commanded_joint(
        node,
        "rail_j1",
        target,
        config.search.rail_joint_tolerance,
        starting_position=starting_position,
        timeout_sec=timeout_sec,
    ):
        raise RuntimeConfigurationError(
            f"rail_j1 did not converge to {target:.4f}m before timeout"
        )


def _command_wrist_and_wait(node: object, target: float, config: RuntimeConfig) -> None:
    starting_position = getattr(node, "current_panda_joint1", None)
    node.send_panda_joint1_command(float(target))
    if not _wait_for_commanded_joint(
        node,
        "panda_joint1",
        target,
        config.search.wrist_joint_tolerance,
        starting_position=starting_position,
    ):
        raise RuntimeConfigurationError(
            f"panda_joint1 did not converge to {target:.4f}rad before timeout"
        )


def _command_default_standing_posture_and_wait(
    node: object, config: RuntimeConfig
) -> None:
    """Move the arm to the configured default standing posture and confirm it."""
    _command_full_arm_posture_and_wait(
        node, STARTUP_ARM_JOINTS, config, label="default standing"
    )


def _command_full_arm_posture_and_wait(
    node: object,
    targets: tuple[float, float, float, float, float, float, float],
    config: RuntimeConfig,
    *,
    label: str,
) -> None:
    """Retry a full-arm direct target until simulator feedback confirms it."""
    send_posture = getattr(node, "send_panda_search_posture", None)
    if not callable(send_posture):
        raise RuntimeConfigurationError(
            "Robot interface does not support full-arm direct-posture commands"
        )

    ensure_rrt_idle = getattr(node, "ensure_rrt_idle", None)
    if callable(ensure_rrt_idle) and not ensure_rrt_idle(
        config.search.targeted_cancel_timeout_sec
    ):
        raise RuntimeConfigurationError(
            f"Refusing {label} posture: active RRT motion could not be stopped"
        )
    ownership_deadline = time.monotonic() + config.search.targeted_cancel_timeout_sec
    while getattr(node, "cartesian_control_active", False) and time.monotonic() < ownership_deadline:
        time.sleep(JOINT_POLL_INTERVAL_SEC)
    if getattr(node, "cartesian_control_active", False):
        raise RuntimeConfigurationError(
            f"Refusing {label} posture while a Cartesian motion owns the arm"
        )

    deadline = time.monotonic() + HOME_MOTION_TIMEOUT_SEC
    next_publish_time = 0.0
    while time.monotonic() < deadline:
        positions = getattr(node, "current_joint_positions", {})
        if all(
            joint_name in positions
            and abs(float(positions[joint_name]) - target)
            <= config.search.wrist_joint_tolerance
            for joint_index, target in enumerate(targets, start=1)
            for joint_name in (f"panda_joint{joint_index}",)
        ):
            # Simulator joint feedback reaches this node and the bridge on
            # separate callbacks.  The bridge needs several stable samples
            # before it releases direct-arm ownership.  Do not publish the
            # following RRT pose into that short window: the bridge would
            # correctly reject it as Cartesian-vs-direct contention.
            release_deadline = (
                time.monotonic() + config.search.targeted_cancel_timeout_sec
            )
            while (
                getattr(node, "direct_joint_control_active", False)
                and time.monotonic() < release_deadline
            ):
                time.sleep(JOINT_POLL_INTERVAL_SEC)
            if getattr(node, "direct_joint_control_active", False):
                raise RuntimeConfigurationError(
                    f"Bridge did not release direct-arm ownership after {label} posture"
                )
            time.sleep(DIRECT_ARM_POSTURE_HANDOFF_DELAY_SEC)
            return
        if time.monotonic() >= next_publish_time:
            send_posture(*targets)
            next_publish_time = time.monotonic() + 0.5
        time.sleep(JOINT_POLL_INTERVAL_SEC)
    raise RuntimeConfigurationError(
        f"Arm did not reach {label} posture before timeout"
    )


def _command_side_manipulation_posture_and_wait(
    node: object, item: dict[str, object], config: RuntimeConfig
) -> None:
    """Set the fixed pre-hover arm posture for the object's table side.

    The two tables lie on opposite global Y sides.  Joint 1 selects the
    matching approach direction while joints 2..7 remain at their initial
    standing angles.
    """
    position = item.get("position")
    if not isinstance(position, dict):
        raise RuntimeConfigurationError("Missing semantic position for side posture")
    joint1 = 1.56 if float(position["y"]) < 0.0 else -1.56
    targets = (joint1, *STARTUP_ARM_JOINTS[1:])
    print(
        "[MANIPULATION][PRE_HOVER_POSTURE] "
        f"table_y={float(position['y']):.3f}, panda_joint1={joint1:.2f}rad"
    )
    _command_full_arm_posture_and_wait(
        node, targets, config, label="pre-hover"
    )


def _return_targeted_search_to_home(node: object, config: RuntimeConfig) -> None:
    """Put the arm in its default posture before returning the rail home."""
    _command_default_standing_posture_and_wait(node, config)
    _command_rail_and_wait(
        node, STARTUP_RAIL_POSITION_M, config, timeout_sec=HOME_MOTION_TIMEOUT_SEC
    )


def _recover_rrt_navigation(node: object, config: RuntimeConfig, *, context: str) -> str | None:
    """Cancel RRT and return the arm to a safe standing posture."""
    try:
        if not node.ensure_rrt_idle(config.search.targeted_cancel_timeout_sec):
            raise RuntimeConfigurationError("RRT did not acknowledge cancellation")
        _command_default_standing_posture_and_wait(node, config)
        print(f"[RECOVERY][{context}] RRT cancelled and arm returned to standing posture.")
        return None
    except Exception as exc:
        message = str(exc)
        print(f"[RECOVERY][{context}] Safe standing-posture recovery failed: {message}")
        return message


def _wait_for_commanded_joint(
    node: object,
    joint_name: str,
    target: float,
    tolerance: float,
    *,
    starting_position: object,
    timeout_sec: float = JOINT_MOTION_TIMEOUT_SEC,
) -> bool:
    """Wait for target convergence and observable progress on small commands."""
    attribute_name = (
        "current_rail_position"
        if joint_name == "rail_j1"
        else "current_panda_joint1"
    )
    start_value = (
        float(starting_position) if starting_position is not None else None
    )
    commanded_distance = (
        abs(target - start_value) if start_value is not None else 0.0
    )
    required_progress = commanded_distance / 2.0
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        value = getattr(node, attribute_name, None)
        if value is not None:
            current = float(value)
            at_target = abs(current - target) <= tolerance
            made_progress = (
                start_value is None
                or commanded_distance <= 1e-12
                or abs(current - start_value) >= required_progress
            )
            # A command may be issued while the joint is already within the
            # requested tolerance (for example startup rail_j1=-1.1011m for
            # a -1.1000m target).  In that case the simulator may correctly
            # hold still, so requiring observable movement would turn a
            # successful command into a timeout.
            if at_target and (commanded_distance <= tolerance or made_progress):
                return True
        time.sleep(JOINT_POLL_INTERVAL_SEC)
    return False


def _joint_is_at_target(
    node: object, joint_name: str, target: float, tolerance: float
) -> bool:
    attribute_name = (
        "current_rail_position"
        if joint_name == "rail_j1"
        else "current_panda_joint1"
    )
    value = getattr(node, attribute_name, None)
    return value is not None and abs(float(value) - target) <= tolerance


def _capture_stationary_detection(
    node: object,
    config: RuntimeConfig,
    target: str,
    *,
    expected_rail: float,
    expected_wrist: float,
) -> dict[str, object]:
    """Capture a fresh frame only after both commanded joints are stationary."""
    if not _wait_for_stationary_joints(
        node,
        config,
        expected_rail=expected_rail,
        expected_wrist=expected_wrist,
    ):
        raise RuntimeConfigurationError(
            "Refusing YOLO inference because the joints did not remain stationary "
            "throughout the settling interval"
        )

    # Discard any image cached before or during motion. The next callback now
    # supplies a frame captured after convergence and the settling interval.
    node.latest_b64_image = None
    return detect_target_in_latest_frame(target)


def _wait_for_stationary_joints(
    node: object,
    config: RuntimeConfig,
    *,
    expected_rail: float,
    expected_wrist: float,
) -> bool:
    """Require both joints to remain converged and stable for the settling period."""
    settling_sec = config.search.motion_settling_sec
    deadline = time.monotonic() + JOINT_MOTION_TIMEOUT_SEC + settling_sec
    stable_since: float | None = None
    previous_rail: float | None = None
    previous_wrist: float | None = None
    rail_stability_tolerance = config.search.rail_joint_tolerance / 4.0
    wrist_stability_tolerance = config.search.wrist_joint_tolerance / 4.0

    while time.monotonic() < deadline:
        rail_value = getattr(node, "current_rail_position", None)
        wrist_value = getattr(node, "current_panda_joint1", None)
        if rail_value is None or wrist_value is None:
            stable_since = None
            time.sleep(JOINT_POLL_INTERVAL_SEC)
            continue

        rail = float(rail_value)
        wrist = float(wrist_value)
        at_targets = _joint_is_at_target(
            node,
            "rail_j1",
            expected_rail,
            config.search.rail_joint_tolerance,
        ) and _joint_is_at_target(
            node,
            "panda_joint1",
            expected_wrist,
            config.search.wrist_joint_tolerance,
        )
        stable_sample = (
            previous_rail is not None
            and previous_wrist is not None
            and abs(rail - previous_rail) <= rail_stability_tolerance
            and abs(wrist - previous_wrist) <= wrist_stability_tolerance
        )
        now = time.monotonic()
        if at_targets and (stable_sample or settling_sec == 0):
            if stable_since is None:
                stable_since = now
            if now - stable_since >= settling_sec:
                return True
        else:
            stable_since = None

        previous_rail = rail
        previous_wrist = wrist
        time.sleep(JOINT_POLL_INTERVAL_SEC)
    return False




def _decelerate_to_halt(node: object, steps: int = 5, step_delay: float = 0.06) -> None:
    """Gracefully ramp rail and j6 velocities to zero over a few steps."""
    for i in range(steps, 0, -1):
        frac = i / steps
        if node.current_panda_joint6 is not None:
            node.send_rail_and_joint6_command(
                rail_val=node.current_rail_position,
                rail_speed=0.0,
                j6_val=node.current_panda_joint6,
                j6_speed=0.0,
            )
        else:
            node.send_absolute_rail_command(node.current_rail_position)
        time.sleep(step_delay)


def _extract_object_world_position(
    node: object,
    bbox: Sequence,
    config: "RuntimeConfig",
) -> dict[str, object]:
    """
    Back-project the YOLO bounding-box centre through the wrist depth image
    into the fixed, rail-aligned global_origin frame.

    Returns a dict with keys: x, y, z, depth_raw, pixel_uv.
    Raises RuntimeConfigurationError on bad depth or TF timeout.
    """
    # ── 1. Grab a fresh depth frame ────────────────────────────────────────
    try:
        depth_img = node.get_latest_depth_image(timeout_sec=3.0)
    except TimeoutError as exc:
        raise RuntimeConfigurationError(f"Depth frame timeout: {exc}") from exc

    h, w = depth_img.shape[:2]

    # ── 2. Compute bounding-box centre pixel ───────────────────────────────
    # bbox format: [x1, y1, x2, y2] in pixels
    u = int((bbox[0] + bbox[2]) / 2.0)
    v = int((bbox[1] + bbox[3]) / 2.0)
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))

    # ── 3. Median depth in a small patch (robust to holes) ─────────────────
    r = config.search.depth_patch_radius
    patch = depth_img[
        max(0, v - r): min(h, v + r + 1),
        max(0, u - r): min(w, u + r + 1),
    ]
    valid = patch[np.isfinite(patch) & (patch > 0.01)]
    if valid.size == 0:
        raise RuntimeConfigurationError(
            f"All depth values in patch around ({u},{v}) are invalid (0 or NaN)."
        )
    depth_m = float(np.median(valid))
    print(f"[POSITION] depth at pixel ({u},{v}): {depth_m:.3f}m")

    # ── 4. Back-project pixel → camera frame (OpenCV convention: Z forward) ─
    fx = config.search.camera_fx
    fy = config.search.camera_fy
    cx = config.search.camera_cx
    cy = config.search.camera_cy
    x_cam = (u - cx) * depth_m / fx
    y_cam = (v - cy) * depth_m / fy
    z_cam = depth_m
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    #print(f"[POSITION] camera frame: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})")

    # ── 5. TF lookup: camera → rail-zero global frame ─────────────────────
    try:
        T_global_cam = node.get_transform_matrix(
            target_frame=config.search.global_origin_frame,
            source_frame=config.search.camera_optical_frame,
            timeout_sec=3.0,
        )
    except TimeoutError as exc:
        raise RuntimeConfigurationError(f"TF lookup failed: {exc}") from exc

    # ── 6. Transform into the static rail-zero global frame ────────────────
    p_global = T_global_cam @ p_cam
    world_x, world_y, world_z = map(float, p_global[:3])

    print(
        f"[POSITION] {config.search.global_origin_frame}: "
        f"({world_x:.3f}, {world_y:.3f}, {world_z:.3f})"
    )

    if not all(math.isfinite(v) for v in (world_x, world_y, world_z)):
        raise RuntimeConfigurationError(
            f"Computed global position contains non-finite values: ({world_x}, {world_y}, {world_z})"
        )

    return {
        "x": world_x,
        "y": world_y,
        "z": world_z,
        "depth_raw": depth_m,
        "pixel_uv": [u, v],
    }


def _persist_dynamic_coordinate_3d(path: Path, target: str, x: float, y: float, z: float) -> None:
    """Write or overwrite a target's {x, y, z} entry in the JSON coordinate file."""
    for val, name in ((x, "x"), (y, "y"), (z, "z")):
        if not math.isfinite(val):
            raise RuntimeConfigurationError(f"Refusing to persist non-finite {name}={val}")
    if not path.parent.is_dir():
        raise RuntimeConfigurationError(
            f"Dynamic-coordinate parent directory does not exist: {path.parent}"
        )

    coordinates: dict[str, object] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeConfigurationError(
                f"Could not read dynamic-coordinate JSON {path}: {exc}"
            ) from exc
        if not isinstance(existing, dict):
            raise RuntimeConfigurationError(
                f"Dynamic-coordinate JSON must contain an object: {path}"
            )
        coordinates.update(existing)

    coordinates[target] = {"x": round(float(x), 4), "y": round(float(y), 4), "z": round(float(z), 4)}

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            temporary_path = Path(tmp.name)
            json.dump(coordinates, tmp, indent=4, sort_keys=True)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeConfigurationError(
            f"Could not persist dynamic coordinate to {path}: {exc}"
        ) from exc


def _semantic_registry_objects(config: RuntimeConfig) -> list[dict[str, object]]:
    """Read the durable semantic-perception registry without changing it."""
    path = config.paths.semantic_objects
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeConfigurationError(
            f"Could not read semantic registry {path}: {exc}"
        ) from exc
    objects = payload.get("objects", []) if isinstance(payload, dict) else []
    if not isinstance(objects, list):
        raise RuntimeConfigurationError("semantic_objects.json has an invalid objects list")
    return [item for item in objects if isinstance(item, dict)]


def _confirmed_semantic_object(
    config: RuntimeConfig, target: str
) -> dict[str, object] | None:
    """Return the most reliable current registry entry for one canonical class."""
    canonical = _normalize_label(target)
    matches: list[dict[str, object]] = []
    for item in _semantic_registry_objects(config):
        position = item.get("position")
        if (
            _normalize_label(item.get("class_name", "")) != canonical
            or item.get("state") != "confirmed"
            or not isinstance(position, dict)
        ):
            continue
        try:
            xyz = tuple(float(position[axis]) for axis in ("x", "y", "z"))
            confidence = float(item.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (*xyz, confidence)):
            continue
        matches.append(item)
    return max(
        matches,
        key=lambda item: (float(item.get("confidence", 0.0)), str(item.get("last_seen", ""))),
        default=None,
    )


def _confirmed_semantic_object_by_id(
    config: RuntimeConfig, object_id: str
) -> dict[str, object] | None:
    """Resolve one exact, usable semantic object instance."""
    requested_id = str(object_id).strip()
    if not requested_id:
        return None
    for item in _semantic_registry_objects(config):
        if str(item.get("id", "")) != requested_id or item.get("state") != "confirmed":
            continue
        position = item.get("position")
        if not isinstance(position, dict):
            continue
        try:
            values = tuple(float(position[axis]) for axis in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values):
            return item
    return None


def get_semantic_objects(class_names: Sequence[str] | None = None) -> dict[str, object]:
    """Return only confirmed requested classes from the durable semantic map.

    An omitted or empty class list is intentionally rejected so an accidental
    full-map response cannot consume the VLM context window.
    """
    try:
        config = get_runtime_config()
        raw_classes = [class_names] if isinstance(class_names, str) else (class_names or [])
        requested = sorted(
            {_normalize_label(name) for name in raw_classes if _normalize_label(name)}
        )
        if not requested:
            return {
                "status": "error",
                "reason": "class_names must contain at least one requested class.",
                "objects": [],
            }
        requested_set = set(requested)
        objects = _semantic_registry_objects(config)
        summary = []
        for item in objects:
            if (
                _normalize_label(item.get("class_name", "")) not in requested_set
                or item.get("state") != "confirmed"
            ):
                continue
            position = item.get("position", {})
            if not isinstance(position, dict):
                continue
            summary.append(
                {
                    "object_id": item.get("id"),
                    "class_name": item.get("class_name"),
                    "state": item.get("state"),
                    "confidence": item.get("confidence"),
                    "x": position.get("x"),
                    "y": position.get("y"),
                    "z": position.get("z"),
                    "last_seen": item.get("last_seen"),
                }
            )
        return {
            "status": "success",
            "path": str(config.paths.semantic_objects),
            "requested_classes": requested,
            "classes_without_match": sorted(
                requested_set
                - {_normalize_label(item["class_name"]) for item in summary}
            ),
            "objects": summary,
        }
    except RuntimeConfigurationError as exc:
        return {"status": "error", "reason": str(exc), "objects": []}


def search_semantic_objects(query: str) -> dict[str, object]:
    """Rank confirmed semantic tracks using a natural-language MobileCLIP query."""
    cleaned = str(query).strip()
    if not cleaned:
        return {
            "status": "error",
            "reason": "query must be a non-empty natural-language object description.",
            "objects": [],
        }
    try:
        response = get_shared_node().search_semantic_objects(
            cleaned,
            timeout_sec=5.0,
            max_results=0,
            minimum_cosine_similarity=-1.0,
        )
    except (RuntimeError, TimeoutError) as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "query": cleaned,
            "objects": [],
        }
    if not response.success:
        return {
            "status": "unavailable",
            "reason": str(response.reason),
            "query": cleaned,
            "appearance_provider_id": str(response.appearance_provider_id),
            "compatible_track_count": int(response.compatible_track_count),
            "objects": [],
        }
    objects = [
        {
            "object_id": match.object_id,
            "class_name": match.class_name,
            "state": match.state,
            "cosine_similarity": round(float(match.cosine_similarity), 6),
            "confidence": round(float(match.confidence), 6),
            "x": round(float(match.position.x), 6),
            "y": round(float(match.position.y), 6),
            "z": round(float(match.position.z), 6),
            "position_stddev_m": round(float(match.position_stddev_m), 6),
            "last_seen": {
                "sec": int(match.last_seen.sec),
                "nanosec": int(match.last_seen.nanosec),
            },
        }
        for match in response.matches
    ]
    return {
        "status": "success",
        "query": cleaned,
        "appearance_provider_id": str(response.appearance_provider_id),
        "compatible_track_count": int(response.compatible_track_count),
        "minimum_cosine_similarity": round(
            float(response.applied_minimum_cosine_similarity), 6
        ),
        "objects": objects,
    }


def _execute_mapping_pose(node: object, pose: object, config: RuntimeConfig) -> None:
    """Execute one mapping viewpoint without target-specific inference or cancellation."""
    node.begin_eef_pose(
        pose.x,
        pose.y,
        pose.z,
        pose.roll,
        pose.pitch,
        pose.yaw,
        readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
        tf_timeout_sec=config.cartesian.tf_timeout_sec,
        base_frame=config.cartesian.base_frame,
        eef_frame=config.cartesian.eef_frame,
    )
    if not node.wait_for_eef_motion(config.cartesian.command_timeout_sec):
        raise RuntimeConfigurationError("RRT mapping viewpoint did not complete successfully")


def _scan_mapping_desk_sides(
    node: object, config: RuntimeConfig, station: float
) -> int:
    """Scan both desk sides from one rail station with semantic perception running."""
    completed = 0
    viewpoints = config.search.mapping_scan_viewpoints
    for side, label in ((1, "left"), (-1, "right")):
        if completed:
            _command_default_standing_posture_and_wait(node, config)
        arc_poses = generate_desk_arc(
            current_rail=station,
            rail_min=config.search.rail_min_position,
            rail_max=config.search.rail_max_position,
            side=side,
            desk_width=config.search.targeted_desk_width,
            radius=config.search.targeted_arc_radius,
            height=config.search.targeted_scan_height,
            roll=config.search.targeted_scan_roll,
            pitch=config.search.targeted_scan_pitch,
            viewpoints=viewpoints,
        )
        targets = generate_desk_targets(
            current_rail=station,
            rail_min=config.search.rail_min_position,
            rail_max=config.search.rail_max_position,
            side=side,
            table_scan_y=config.search.targeted_table_scan_y,
            surface_z=config.search.targeted_desk_surface_z,
            viewpoints=viewpoints,
        )
        print(f"[MAPPING][{label.upper()}] {viewpoints} RRT viewpoints at rail={station:.3f}m")
        for index, (arc_pose, target_point) in enumerate(zip(arc_poses, targets, strict=True), 1):
            pose = _camera_look_at_scan_pose(node, arc_pose, target_point, config)
            print(f"[MAPPING][{label.upper()}] viewpoint {index}/{viewpoints}")
            _execute_mapping_pose(node, pose, config)
            completed += 1
    return completed


def general_mapping() -> dict[str, object]:
    """Build a full semantic map from min, centre, and max rail stations."""
    completed_viewpoints = 0
    started = time.monotonic()
    node = None
    config = None
    try:
        config = get_runtime_config()
        node = get_shared_node()
        if not _wait_for_search_telemetry(node):
            return _failure_result("", "Timed out waiting for rail and arm telemetry.", state="initialize")
        stations = (
            config.search.rail_min_position,
            rail_centre(config.search.rail_min_position, config.search.rail_max_position),
            config.search.rail_max_position,
        )
        for index, station in enumerate(stations, 1):
            print(f"[MAPPING][STATION] {index}/3: moving rail to {station:.3f}m")
            _command_default_standing_posture_and_wait(node, config)
            _command_rail_and_wait(node, station, config)
            completed_viewpoints += _scan_mapping_desk_sides(node, config, station)
        _return_targeted_search_to_home(node, config)
        return {
            "status": "success",
            "success": True,
            "state": "complete",
            "stations": list(stations),
            "viewpoints_completed": completed_viewpoints,
            "viewpoints_planned": 6 * config.search.mapping_scan_viewpoints,
            "elapsed_sec": round(time.monotonic() - started, 2),
        }
    except Exception as exc:
        print(f"[MAPPING][FAILURE] {exc}")
        recovery_error = None
        if node is not None and config is not None:
            recovery_error = _recover_rrt_navigation(node, config, context="MAPPING")
        reason = f"Mapping aborted safely: {exc}"
        if recovery_error:
            reason += f"; standing-posture recovery failed: {recovery_error}"
        return _failure_result(
            "", reason, state="failure", observations=completed_viewpoints,
            recoverable=True,
        )


def _legacy_general_mapping(target_object: str) -> dict[str, object]:
    """General mapping tool: full rail sweep with YOLO to locate and persist target coordinates."""
    target = _normalize_label(target_object or "")
    if not target:
        return _failure_result(
            target,
            "A non-empty canonical target label is required.",
            state="initialize",
        )

    observations = 0
    try:
        config = get_runtime_config()
        
        if config.paths.static_semantic_coordinates.is_file():
            try:
                with config.paths.static_semantic_coordinates.open("r", encoding="utf-8") as f:
                    known_distances = json.load(f)
                if target in known_distances:
                    print(f"[SEARCH SKIP] Target '{target}' already known in semantic map. Skipping active YOLO search.")
                    return {
                        "status": "success",
                        "success": True,
                        "target": target,
                        "reason": "Found in static semantic map",
                        "state": "complete",
                    }
            except (OSError, json.JSONDecodeError) as e:
                print(f"[WARNING] Could not read static semantic map: {e}. Falling back to active search.")
        detector = _get_yolo_detector(config)
        if not detector.supports_label(target):
            return _failure_result(
                target,
                f"Configured YOLO checkpoint does not provide target class {target!r}.",
                state="initialize",
            )

        node = get_shared_node()
        if not _wait_for_search_telemetry(node):
            return _failure_result(
                target,
                "Timed out waiting for rail and wrist telemetry.",
                state="initialize",
            )

        current_rail_position = float(node.current_rail_position)
        if not math.isfinite(current_rail_position):
            return _failure_result(
                target,
                "Rail telemetry is not finite.",
                state="initialize",
            )
        limits = [config.search.rail_min_position, config.search.rail_max_position]
        start_limit = limits[0] if abs(current_rail_position - limits[0]) < abs(current_rail_position - limits[1]) else limits[1]

        print(
            f"[SEARCH][INITIALIZE] target={target!r}, "
            f"rail={current_rail_position:.4f}m, "
            f"frame={config.search.global_origin_frame!r}."
        )
        angles_to_scan = [angle for angle in config.search.wrist_search_angles if abs(angle) > 0.1]
        if not angles_to_scan:
            angles_to_scan = [1.57, -1.57]

        target_centered = False
        final_detection = None

        print(f"[SEARCH][INITIALIZE] Moving to start limit {start_limit:.4f}m...")
        _command_rail_and_wait(node, start_limit, config)
        
        current_limit = start_limit

        for sweep_idx, wrist_angle in enumerate(angles_to_scan):
            if target_centered:
                break
                
            target_limit = limits[1] if current_limit == limits[0] else limits[0]
            
            j6_limits = [config.search.j6_min, config.search.j6_max]
            j6_target = j6_limits[1]
            
            if sweep_idx == 0:
                print(f"[SEARCH][POSTURE] Setting initial arm search posture (j1={wrist_angle}, j2={config.search.search_posture_j2}, j3={config.search.search_posture_j3}, j4={config.search.search_posture_j4}, j5={config.search.search_posture_j5}, j7={config.search.search_posture_j7})...")
                if hasattr(node, "send_panda_search_posture"):
                    current_j6 = node.current_panda_joint6 if node.current_panda_joint6 is not None else j6_limits[0]
                    node.send_panda_search_posture(wrist_angle, config.search.search_posture_j2, config.search.search_posture_j3, config.search.search_posture_j4, config.search.search_posture_j5, current_j6, config.search.search_posture_j7)
                else:
                    _command_wrist_and_wait(node, wrist_angle, config)
            else:
                print(f"[SEARCH][TURN] Turning to j1={wrist_angle}, freezing all other joints.")
                node.send_panda_joint1_command(wrist_angle)
            
            print("[SEARCH][WAIT] Waiting 5.0 seconds for arm posture/turn to settle...")
            time.sleep(7.0) 
            
            # Start oscillating j6 and moving rail simultaneously
            print(f"[SEARCH][RAIL] Sweep {sweep_idx+1}/{len(angles_to_scan)} - moving continuously to {target_limit:.4f}m...")
            
            rail_dir = 1.0 if target_limit >= node.current_rail_position else -1.0
            j6_dir = 1.0 if j6_target >= (node.current_panda_joint6 if node.current_panda_joint6 is not None else j6_limits[0]) else -1.0
            
            node.send_rail_and_joint6_command(
                rail_val=target_limit, 
                rail_speed=config.search.rail_speed * rail_dir, 
                j6_val=j6_target, 
                j6_speed=config.search.j6_speed * j6_dir
            )

            while True:
                # Handle j6 oscillation
                if node.current_panda_joint6 is not None:
                    if abs(node.current_panda_joint6 - j6_target) < 0.15:
                        j6_target = j6_limits[0] if j6_target == j6_limits[1] else j6_limits[1]
                        print(f"[SEARCH][J6] Flipped target to {j6_target:.2f}rad")
                        
                        rail_dir = 1.0 if target_limit >= node.current_rail_position else -1.0
                        j6_dir = 1.0 if j6_target >= node.current_panda_joint6 else -1.0
                        
                        node.send_rail_and_joint6_command(
                            rail_val=target_limit,
                            rail_speed=config.search.rail_speed * rail_dir,
                            j6_val=j6_target,
                            j6_speed=config.search.j6_speed * j6_dir
                        )
                        
                node.latest_b64_image = None
                try:
                    _ = get_latest_ros_image(timeout_sec=2.0)
                except TimeoutError:
                    pass
                
                detection = detect_target_in_latest_frame(target, timeout_sec=1.0)
                observations += 1
                
                # Only halt the fast sweep if confidence is at least 0.5
                is_confident = detection.get("confidence") is not None and float(detection["confidence"]) >= 0.5
                
                if detection["status"] == "not_found" or not is_confident:
                    if abs(node.current_rail_position - target_limit) < config.search.rail_joint_tolerance:
                        print(f"[SEARCH][RAIL] Reached limit {target_limit:.4f}m. Halting motion before next sweep.")
                        
                        # Explicitly halt both rail and tilt
                        if node.current_panda_joint6 is not None:
                            node.send_rail_and_joint6_command(
                                rail_val=node.current_rail_position,
                                rail_speed=0.0,
                                j6_val=node.current_panda_joint6,
                                j6_speed=0.0
                            )
                        else:
                            node.send_absolute_rail_command(node.current_rail_position)
                            
                        current_limit = target_limit
                        break
                    continue

                print(
                    "[SEARCH][DETECTED] Candidate found; gracefully decelerating and "
                    "extracting object position."
                )

                # Graceful deceleration: ramp velocity to zero over ~0.3s
                _decelerate_to_halt(node)

                # Extract 3-D rail-zero global position from the depth frame
                try:
                    position = _extract_object_world_position(node, detection["bbox"], config)
                except RuntimeConfigurationError as pos_exc:
                    print(f"[SEARCH][POSITION] Position extraction failed: {pos_exc}. Resuming sweep.")
                    # Resume sweep
                    rail_dir = 1.0 if target_limit >= node.current_rail_position else -1.0
                    j6_dir = 1.0 if j6_target >= (node.current_panda_joint6 or j6_limits[0]) else -1.0
                    node.send_rail_and_joint6_command(
                        rail_val=target_limit,
                        rail_speed=config.search.rail_speed * rail_dir,
                        j6_val=j6_target,
                        j6_speed=config.search.j6_speed * j6_dir,
                    )
                    continue

                # Persist x, y, z to the JSON coordinate file
                try:
                    _persist_dynamic_coordinate_3d(
                        config.paths.dynamic_semantic_coordinates,
                        target,
                        position["x"],
                        position["y"],
                        position["z"],
                    )
                except RuntimeConfigurationError as exc:
                    return _failure_result(
                        target,
                        str(exc),
                        state="persist_coordinate",
                        detection=detection,
                        observations=observations,
                    )

                target_centered = True
                return {
                    "status": "success",
                    "success": True,
                    "state": "save_and_succeed",
                    "reason": None,
                    "target": target,
                    "confidence": detection["confidence"],
                    "bbox": detection["bbox"],
                    "center": detection.get("center"),
                    "x": position["x"],
                    "y": position["y"],
                    "z": position["z"],
                    "depth_m": position["depth_raw"],
                    "pixel_uv": position["pixel_uv"],
                    "rail_position": float(node.current_rail_position),
                    "observations": observations,
                    "coordinate_path": str(config.paths.dynamic_semantic_coordinates),
                }

        if not target_centered:
            return _failure_result(
                target,
                "Full configured rail range was scanned without a valid detection.",
                state="search_sweep",
                observations=observations,
            )

        # Should not reach here; belt-and-suspenders
        return _failure_result(
            target,
            "Search concluded without finding the target.",
            state="failure",
            observations=observations,
        )
    except Exception as exc:
        return _failure_result(
            target,
            f"Search aborted safely: {exc}",
            state="failure",
            observations=observations,
        )
        

def _scan_pose_with_yolo(
    node: object,
    target: str,
    pose: object,
    config: RuntimeConfig,
) -> tuple[dict[str, object] | None, int]:
    """Run one RRT arc segment while sampling YOLO at configured FPS."""
    period = 1.0 / config.search.targeted_capture_fps
    node.begin_eef_pose(
        pose.x,
        pose.y,
        pose.z,
        pose.roll,
        pose.pitch,
        pose.yaw,
        readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
        tf_timeout_sec=config.cartesian.tf_timeout_sec,
        base_frame=config.cartesian.base_frame,
        eef_frame=config.cartesian.eef_frame,
    )
    observations = 0
    next_capture = time.monotonic()
    first_observation = True
    while first_observation or not node.eef_motion_done():
        first_observation = False
        now = time.monotonic()
        if now < next_capture:
            time.sleep(min(next_capture - now, 0.01))
            continue

        started = time.monotonic()
        node.latest_b64_image = None
        try:
            detection = detect_target_in_latest_frame(target, timeout_sec=period)
        except Exception as exc:
            cancelled = node.cancel_eef_motion(
                config.search.targeted_cancel_timeout_sec
            )
            suffix = "" if cancelled else "; RRT cancellation also timed out"
            raise RuntimeConfigurationError(
                f"Targeted frame delivery/inference failed: {exc}{suffix}"
            ) from exc
        observations += 1
        elapsed = time.monotonic() - started
        if elapsed > period:
            cancelled = node.cancel_eef_motion(
                config.search.targeted_cancel_timeout_sec
            )
            if not cancelled:
                raise RuntimeConfigurationError(
                    "YOLO deadline was missed and RRT cancellation timed out"
                )
            raise RuntimeConfigurationError(
                f"YOLO delivery deadline missed: {elapsed:.3f}s > {period:.3f}s "
                f"({config.search.targeted_capture_fps:g} FPS)"
            )

        confidence = detection.get("confidence")
        if (
            detection.get("status") != "not_found"
            and confidence is not None
            and float(confidence) >= config.search.targeted_candidate_confidence
        ):
            if not node.cancel_eef_motion(config.search.targeted_cancel_timeout_sec):
                raise RuntimeConfigurationError("Timed out cancelling RRT after detection")
            time.sleep(config.search.motion_settling_sec)
            return detection, observations

        next_capture += period
        if time.monotonic() > next_capture:
            cancelled = node.cancel_eef_motion(
                config.search.targeted_cancel_timeout_sec
            )
            if not cancelled:
                raise RuntimeConfigurationError(
                    "Targeted schedule slipped and RRT cancellation timed out"
                )
            raise RuntimeConfigurationError(
                f"Targeted capture schedule missed "
                f"{config.search.targeted_capture_fps:g} FPS"
            )

    if not node.wait_for_eef_motion(0.0):
        raise RuntimeConfigurationError("RRT scan pose did not complete successfully")
    return None, observations


def _camera_look_at_scan_pose(
    node: object,
    pose: object,
    target: object,
    config: RuntimeConfig,
) -> object:
    """Keep the arc XYZ while orienting the wrist camera at one desk point."""
    try:
        base_to_eef = node.get_transform_matrix(
            target_frame=config.cartesian.base_frame,
            source_frame=config.cartesian.eef_frame,
            timeout_sec=config.cartesian.tf_timeout_sec,
        )
        base_to_camera = node.get_transform_matrix(
            target_frame=config.cartesian.base_frame,
            source_frame=config.search.camera_optical_frame,
            timeout_sec=config.cartesian.tf_timeout_sec,
        )
        eef_to_camera = np.linalg.inv(base_to_eef) @ base_to_camera
        eef_rotation = look_at_eef_rotation(
            eef_position=(pose.x, pose.y, pose.z),
            target_position=(target.x, target.y, target.z),
            eef_to_camera=eef_to_camera,
        )
        roll, pitch, yaw = Rotation.from_matrix(eef_rotation).as_euler("xyz")
    except (TimeoutError, ValueError, np.linalg.LinAlgError) as exc:
        raise RuntimeConfigurationError(
            f"Could not compute camera look-at scan orientation: {exc}"
        ) from exc

    if not all(math.isfinite(float(value)) for value in (roll, pitch, yaw)):
        raise RuntimeConfigurationError("Camera look-at produced non-finite EEF RPY")
    roll = float(roll)
    return type(pose)(
        x=pose.x,
        y=pose.y,
        z=pose.z,
        roll=roll,
        pitch=float(pitch),
        yaw=float(yaw),
    )


def _scan_both_desk_sides(
    node: object,
    target: str,
    config: RuntimeConfig,
) -> tuple[dict[str, object] | None, int | None, int]:
    observations = 0
    current_rail = float(node.current_rail_position)
    for idx, (side, label) in enumerate(((1, "left"), (-1, "right"))):
        if idx > 0:
            print(f"[TARGETED][RRT] Returning to initial posture before scanning {label.upper()} side...")
            try:
                _command_default_standing_posture_and_wait(node, config)
            except RuntimeConfigurationError as e:
                print(f"[TARGETED][RRT] Failed to return to initial posture: {e}")
        arc_poses = generate_desk_arc(
            current_rail=current_rail,
            rail_min=config.search.rail_min_position,
            rail_max=config.search.rail_max_position,
            side=side,
            desk_width=config.search.targeted_desk_width,
            radius=config.search.targeted_arc_radius,
            height=config.search.targeted_scan_height,
            roll=config.search.targeted_scan_roll,
            pitch=config.search.targeted_scan_pitch,
            viewpoints=config.search.targeted_scan_viewpoints,
        )
        targets = generate_desk_targets(
            current_rail=current_rail,
            rail_min=config.search.rail_min_position,
            rail_max=config.search.rail_max_position,
            side=side,
            table_scan_y=config.search.targeted_table_scan_y,
            surface_z=config.search.targeted_desk_surface_z,
            viewpoints=config.search.targeted_scan_viewpoints,
        )
        print(
            f"[TARGETED][{label.upper()}] Scanning {len(arc_poses)} RRT viewpoints "
            f"at {config.search.targeted_capture_fps:g} FPS."
        )
        for index, (arc_pose, target_point) in enumerate(
            zip(arc_poses, targets, strict=True), start=1
        ):
            pose = _camera_look_at_scan_pose(node, arc_pose, target_point, config)
            print(f"[TARGETED][{label.upper()}] viewpoint {index}/{len(arc_poses)}")
            detection, count = _scan_pose_with_yolo(node, target, pose, config)
            observations += count
            if detection is not None:
                return detection, side, observations
    return None, None, observations


def _targeted_search_legacy_local_yolo(target_object: str) -> dict[str, object]:
    """Search both desk rows with RRT wrist arcs and confirm candidates up close."""
    target = _normalize_label(target_object or "")
    if not target:
        return _failure_result(
            target, "A non-empty target label is required.", state="initialize"
        )

    observations = 0
    try:
        config = get_runtime_config()
        detector = _get_yolo_detector(config)
        if not detector.supports_label(target):
            return _failure_result(
                target,
                f"Configured YOLO checkpoint does not provide target class {target!r}.",
                state="initialize",
            )
        node = get_shared_node()
        if not _wait_for_search_telemetry(node):
            return _failure_result(
                target,
                "Timed out waiting for rail and arm telemetry.",
                state="initialize",
            )

        centre = rail_centre(
            config.search.rail_min_position, config.search.rail_max_position
        )
        stations = [float(node.current_rail_position)]
        if not math.isclose(
            stations[0], centre, abs_tol=config.search.rail_joint_tolerance
        ):
            stations.append(centre)

        candidate = None
        candidate_side = None
        for station_index, station in enumerate(stations):
            if station_index > 0:
                print("[TARGETED][POSTURE] Returning arm to default standing posture.")
                _command_default_standing_posture_and_wait(node, config)
                print(f"[TARGETED][CENTRE] Moving rail to {station:.3f}m for retry.")
                _command_rail_and_wait(node, station, config)
            candidate, candidate_side, count = _scan_both_desk_sides(
                node, target, config
            )
            observations += count
            if candidate is not None:
                break

        if candidate is None or candidate_side is None:
            print("[TARGETED][HOME] No candidate found; returning to home posture.")
            try:
                _return_targeted_search_to_home(node, config)
            except RuntimeConfigurationError as recovery_exc:
                return _failure_result(
                    target,
                    "Couldn't find target after scanning both desk sides and rail centre; "
                    f"home recovery failed: {recovery_exc}",
                    state="home_recovery",
                    observations=observations,
                )
            return _failure_result(
                target,
                "Couldn't find target after scanning both desk sides and rail centre.",
                state="couldnt_find",
                observations=observations,
            )

        print("[TARGETED][CANDIDATE] Extracting global 3-D position.")
        approximate = _extract_object_world_position(node, candidate["bbox"], config)
        approach_x = max(
            config.search.rail_min_position,
            min(config.search.rail_max_position, float(approximate["x"])),
        )
        approximate_coordinate_path = (
            config.paths.dynamic_semantic_coordinates.with_name(
                "semantic_distances_dynamic.json"
            )
        )
        try:
            _persist_dynamic_coordinate_3d(
                approximate_coordinate_path,
                target + "_approximate",
                approximate["x"],
                approximate["y"],
                approximate["z"],
            )
        except RuntimeConfigurationError as exc:
            # The approximate map is debug-only.  Its write failure must not
            # prevent final localization from continuing.
            print(f"[TARGETED][DEBUG] Couldn't save approximate target: {exc}")

        print("[TARGETED][POSTURE] Returning arm to initial pose before approach.")
        try:
            _command_default_standing_posture_and_wait(node, config)
        except RuntimeConfigurationError as exc:
            return _failure_result(
                target,
                f"Couldn't return arm to initial pose before approaching target: {exc}",
                state="initial_pose_recovery",
                detection=candidate,
                observations=observations,
            )

        print(f"[TARGETED][APPROACH] Moving to target rail position {approach_x:.3f}m.")
        try:
            _command_rail_and_wait(node, approach_x, config)
        except RuntimeConfigurationError as exc:
            return _failure_result(
                target,
                f"Rail approach to target failed: {exc}",
                state="approach_move",
                detection=candidate,
                observations=observations,
            )

        # The rail-global frame and panda_link0 are rotated 180 degrees about
        # Z, so panda_link0 Y is the opposite sign of global_origin Y.  Pick
        # the close-scan table side from the approximate global Y accordingly.
        scan_side = -1 if float(approximate["y"]) >= 0.0 else 1
        close_poses = close_view_tilt_poses(
            side=scan_side,
            standoff=config.search.targeted_close_standoff,
            height=config.search.targeted_scan_height,
            roll=config.search.targeted_scan_roll,
            pitch=config.search.targeted_scan_pitch,
        )
        confirmations: list[tuple[float, dict[str, object], dict[str, object]]] = []
        best_detection = candidate
        best_confidence = float(candidate.get("confidence") or 0.0)
        for tilt_label, close_pose in zip(("down", "up"), close_poses, strict=True):
            print(f"[TARGETED][CONFIRM][{tilt_label.upper()}]")
            try:
                pose_completed = node.send_eef_pose(
                    close_pose.x,
                    close_pose.y,
                    close_pose.z,
                    close_pose.roll,
                    close_pose.pitch,
                    close_pose.yaw,
                    timeout_sec=config.cartesian.command_timeout_sec,
                    readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
                    tf_timeout_sec=config.cartesian.tf_timeout_sec,
                    base_frame=config.cartesian.base_frame,
                    eef_frame=config.cartesian.eef_frame,
                )
            except (RuntimeError, TimeoutError, ValueError) as exc:
                pose_completed = False
                pose_error = str(exc)
            else:
                pose_error = "did not complete"
            if not pose_completed:
                try:
                    _return_targeted_search_to_home(node, config)
                except RuntimeConfigurationError as recovery_exc:
                    return _failure_result(
                        target,
                        f"Closer-look {tilt_label} RRT pose {pose_error}; "
                        f"initial-pose recovery also failed: {recovery_exc}",
                        state="home_recovery",
                        detection=best_detection,
                        observations=observations,
                    )
                return _failure_result(
                    target,
                    f"Closer-look {tilt_label} RRT pose {pose_error}; returned to initial pose.",
                    state="close_approach",
                    detection=best_detection,
                    observations=observations,
                )

            time.sleep(config.search.motion_settling_sec)
            node.latest_b64_image = None
            detection = detect_target_in_latest_frame(
                target, timeout_sec=1.0 / config.search.targeted_capture_fps
            )
            observations += 1
            confidence = detection.get("confidence")
            numeric_confidence = float(confidence) if confidence is not None else 0.0
            if numeric_confidence > best_confidence:
                best_detection = detection
                best_confidence = numeric_confidence
            if (
                detection.get("status") == "not_found"
                or confidence is None
                or numeric_confidence <= config.yolo.confidence_threshold
            ):
                continue
            position = _extract_object_world_position(
                node, detection["bbox"], config
            )
            confirmations.append((numeric_confidence, detection, position))
            print(
                f"[TARGETED][CONFIRM] Target confirmed during {tilt_label} scan. "
                "Continuing scan sequence to gather best candidate."
            )

        if not confirmations:
            print("[TARGETED][HOME] Target not confirmed; returning to initial pose.")
            try:
                _return_targeted_search_to_home(node, config)
            except RuntimeConfigurationError as recovery_exc:
                return _failure_result(
                    target,
                    "Candidate failed both up/down closer-look confirmations; "
                    f"initial-pose recovery failed: {recovery_exc}",
                    state="home_recovery",
                    detection=best_detection,
                    observations=observations,
                )
            return _failure_result(
                target,
                "Candidate failed both up/down closer-look confirmations; returned to initial pose.",
                state="close_confirmation",
                detection=best_detection,
                observations=observations,
            )

        _, final_detection, final_position = max(
            confirmations, key=lambda confirmation: confirmation[0]
        )
        final_rail_position = max(
            config.search.rail_min_position,
            min(config.search.rail_max_position, float(final_position["x"])),
        )
        print(
            "[TARGETED][ALIGN] Keeping confirmation joint pose and moving rail to "
            f"final target X={final_rail_position:.3f}m."
        )
        try:
            _command_rail_and_wait(node, final_rail_position, config)
        except RuntimeConfigurationError as exc:
            return _failure_result(
                target,
                f"Target was confirmed but final rail alignment failed: {exc}",
                state="final_alignment",
                detection=final_detection,
                observations=observations,
            )
        _persist_dynamic_coordinate_3d(
            config.paths.dynamic_semantic_coordinates,
            target,
            final_position["x"],
            final_position["y"],
            final_position["z"],
        )
        return {
            "status": "success",
            "success": True,
            "state": "save_and_succeed",
            "reason": None,
            "target": target,
            "confidence": final_detection["confidence"],
            "bbox": final_detection["bbox"],
            "center": final_detection.get("center"),
            "x": final_position["x"],
            "y": final_position["y"],
            "z": final_position["z"],
            "depth_m": final_position["depth_raw"],
            "pixel_uv": final_position["pixel_uv"],
            "rail_position": float(node.current_rail_position),
            "observations": observations,
            "coordinate_path": str(config.paths.dynamic_semantic_coordinates),
            "approximate_coordinate_path": str(approximate_coordinate_path),
        }
    except Exception as exc:
        return _failure_result(
            target,
            f"Targeted search aborted safely: {exc}",
            state="failure",
            observations=observations,
        )


def _semantic_target_matches(item: object, target: str, minimum: float) -> bool:
    return bool(
        item is not None
        and _normalize_label(getattr(item, "class_name", "")) == target
        and float(getattr(item, "confidence", 0.0)) >= minimum
    )


def _semantic_result_object(node: object, target: str, minimum: float) -> object | None:
    status = node.semantic_find_status()
    if status["error"]:
        raise RuntimeConfigurationError(f"Semantic FindObject failed: {status['error']}")
    result = status["result"]
    if result is not None and bool(result.found):
        item = result.object
        if _semantic_target_matches(item, target, minimum):
            return item
    return None


def _hold_semantic_candidate(
    node: object, candidate: object, target: str, config: RuntimeConfig
) -> object | None:
    """Aim at one candidate until semantic perception confirms or expires it."""
    global_x = float(candidate.position.x)
    global_y = float(candidate.position.y)
    global_z = float(candidate.position.z)
    rail = float(node.current_rail_position)
    side = -1 if global_y >= 0.0 else 1
    pose = ScanPose(
        x=0.0,
        y=side * config.search.targeted_close_standoff,
        z=config.search.targeted_scan_height,
        roll=config.search.targeted_scan_roll,
        pitch=config.search.targeted_scan_pitch,
        yaw=0.0,
    )
    base_target = type("SemanticTarget", (), {
        "x": rail - global_x, "y": -global_y, "z": global_z
    })()
    aimed_pose = _camera_look_at_scan_pose(node, pose, base_target, config)
    print(f"[TARGETED][CANDIDATE] Holding view of {candidate.object_id}.")
    # Cancellation already leaves the robot at the camera pose that detected
    # this candidate.  A closer re-aim is useful when feasible, but it must not
    # turn an unconfirmed candidate into a terminal search failure.
    try:
        hold_completed = node.send_eef_pose(
            aimed_pose.x, aimed_pose.y, aimed_pose.z,
            aimed_pose.roll, aimed_pose.pitch, aimed_pose.yaw,
            timeout_sec=config.cartesian.command_timeout_sec,
            readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
            tf_timeout_sec=config.cartesian.tf_timeout_sec,
            base_frame=config.cartesian.base_frame, eef_frame=config.cartesian.eef_frame,
        )
        if not hold_completed:
            print(
                "[TARGETED][CANDIDATE] Re-aim pose did not complete; "
                "holding the cancelled detection view instead."
            )
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(
            "[TARGETED][CANDIDATE] Re-aim pose was unavailable; "
            f"holding the cancelled detection view instead: {exc}"
        )

    candidate_id = candidate.object_id
    while True:
        confirmed = _semantic_result_object(
            node, target, config.search.targeted_confirmation_confidence
        )
        if confirmed is not None:
            return confirmed
        current = node.semantic_object(candidate_id)
        if current is None and node.semantic_find_status()["objects_received"]:
            print("[TARGETED][CANDIDATE] Candidate expired; resuming scan.")
            return None
        if current is not None and str(current.state) != "candidate":
            print(
                "[TARGETED][CANDIDATE] Candidate did not confirm "
                f"(state={current.state}); resuming scan."
            )
            return None
        if _semantic_target_matches(
            current, target, config.search.targeted_confirmation_confidence
        ) and str(current.state) == "confirmed":
            return current
        status = node.semantic_find_status()
        if status["result"] is not None:
            print("[TARGETED][CANDIDATE] Semantic action completed without confirmation; resuming scan.")
            return None
        time.sleep(0.05)


def _scan_pose_with_semantic(
    node: object, target: str, pose: ScanPose, config: RuntimeConfig
) -> object | None:
    """Run one RRT scan pose and interrupt it only for a semantic candidate."""
    node.begin_eef_pose(
        pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw,
        readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
        tf_timeout_sec=config.cartesian.tf_timeout_sec,
        base_frame=config.cartesian.base_frame, eef_frame=config.cartesian.eef_frame,
    )
    while not node.eef_motion_done():
        confirmed = _semantic_result_object(
            node, target, config.search.targeted_confirmation_confidence
        )
        if confirmed is not None:
            node.cancel_eef_motion(config.search.targeted_cancel_timeout_sec)
            return confirmed
        semantic_status = node.semantic_find_status()
        candidate = semantic_status["candidate"]
        if _semantic_target_matches(
            candidate, target, config.search.targeted_candidate_confidence
        ) and (
            not semantic_status["objects_received"]
            or node.semantic_object(candidate.object_id) is not None
        ):
            if not node.cancel_eef_motion(config.search.targeted_cancel_timeout_sec):
                raise RuntimeConfigurationError("Timed out cancelling RRT for semantic candidate")
            return _hold_semantic_candidate(node, candidate, target, config)
        time.sleep(0.05)
    if not node.wait_for_eef_motion(0.0):
        raise RuntimeConfigurationError("RRT scan pose did not complete successfully")
    return _semantic_result_object(node, target, config.search.targeted_confirmation_confidence)


def _scan_semantic_both_desk_sides(node: object, target: str, config: RuntimeConfig) -> object | None:
    current_rail = float(node.current_rail_position)
    for side, label in ((1, "left"), (-1, "right")):
        arc_poses = generate_desk_arc(
            current_rail=current_rail, rail_min=config.search.rail_min_position,
            rail_max=config.search.rail_max_position, side=side,
            desk_width=config.search.targeted_desk_width,
            radius=config.search.targeted_arc_radius, height=config.search.targeted_scan_height,
            roll=config.search.targeted_scan_roll, pitch=config.search.targeted_scan_pitch,
            viewpoints=config.search.targeted_scan_viewpoints,
        )
        targets = generate_desk_targets(
            current_rail=current_rail, rail_min=config.search.rail_min_position,
            rail_max=config.search.rail_max_position, side=side,
            table_scan_y=config.search.targeted_table_scan_y,
            surface_z=config.search.targeted_desk_surface_z,
            viewpoints=config.search.targeted_scan_viewpoints,
        )
        print(f"[TARGETED][{label.upper()}] Scanning {len(arc_poses)} RRT viewpoints using semantic perception.")
        for arc_pose, target_point in zip(arc_poses, targets, strict=True):
            confirmed = _scan_pose_with_semantic(
                node, target, _camera_look_at_scan_pose(node, arc_pose, target_point, config), config
            )
            if confirmed is not None:
                return confirmed
        if side == 1:
            _command_default_standing_posture_and_wait(node, config)
    return None


def targeted_search(target_object: str) -> dict[str, object]:
    """Search RRT scan paths while semantic perception owns all detection and depth work."""
    target = _normalize_label(target_object or "")
    if not target:
        return _failure_result(target, "A non-empty target label is required.", state="initialize")
    node = None
    config = None
    try:
        config = get_runtime_config()
        node = get_shared_node()
        if not _wait_for_search_telemetry(node):
            return _failure_result(target, "Timed out waiting for rail and arm telemetry.", state="initialize")
        node.start_semantic_find_object(
            target, config.search.targeted_confirmation_confidence,
            config.search.targeted_semantic_action_timeout_sec,
        )
        centre = rail_centre(config.search.rail_min_position, config.search.rail_max_position)
        stations = [float(node.current_rail_position)]
        if not math.isclose(stations[0], centre, abs_tol=config.search.rail_joint_tolerance):
            stations.append(centre)
        confirmed = None
        for index, station in enumerate(stations):
            if index:
                _command_default_standing_posture_and_wait(node, config)
                _command_rail_and_wait(node, station, config)
            confirmed = _scan_semantic_both_desk_sides(node, target, config)
            if confirmed is not None:
                break
        if confirmed is None:
            node.cancel_semantic_find_object()
            _return_targeted_search_to_home(node, config)
            return _failure_result(target, "Couldn't find target after scanning both desk sides and rail centre.", state="couldnt_find")

        # Confirmation is semantic.  Move to its global X, then leave the camera
        # near and aimed at its published global position.
        _command_default_standing_posture_and_wait(node, config)
        final_rail = max(config.search.rail_min_position, min(config.search.rail_max_position, float(confirmed.position.x)))
        _command_rail_and_wait(node, final_rail, config)
        side = -1 if float(confirmed.position.y) >= 0.0 else 1
        pose = ScanPose(0.0, side * config.search.targeted_close_standoff,
                        config.search.targeted_scan_height, config.search.targeted_scan_roll,
                        config.search.targeted_scan_pitch, 0.0)
        base_target = type("SemanticTarget", (), {
            "x": final_rail - float(confirmed.position.x),
            "y": -float(confirmed.position.y), "z": float(confirmed.position.z)
        })()
        pose = _camera_look_at_scan_pose(node, pose, base_target, config)
        if not node.send_eef_pose(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw,
                                  timeout_sec=config.cartesian.command_timeout_sec,
                                  readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
                                  tf_timeout_sec=config.cartesian.tf_timeout_sec,
                                  base_frame=config.cartesian.base_frame, eef_frame=config.cartesian.eef_frame):
            raise RuntimeConfigurationError("Final semantic look-at pose did not complete")
        return {
            "status": "success", "success": True, "state": "semantic_confirmed",
            "reason": None, "target": target, "object_id": confirmed.object_id,
            "confidence": float(confirmed.confidence), "x": float(confirmed.position.x),
            "y": float(confirmed.position.y), "z": float(confirmed.position.z),
            "position_stddev_m": float(confirmed.position_stddev_m),
            "rail_position": float(node.current_rail_position),
            "coordinate_path": str(config.paths.semantic_objects),
        }
    except Exception as exc:
        print(f"[TARGETED][FAILURE] {exc}")
        try:
            get_shared_node().cancel_semantic_find_object()
        except Exception:
            pass
        recovery_error = (
            _recover_rrt_navigation(node, config, context="TARGETED")
            if node is not None and config is not None
            else None
        )
        reason = f"Targeted semantic search aborted safely: {exc}"
        if recovery_error:
            reason += f"; standing-posture recovery failed: {recovery_error}"
        return _failure_result(target, reason, state="failure", recoverable=True)


def get_latest_ros_image(timeout_sec=10.0) -> str:
    node = get_shared_node()
    start = time.time()
    while node.latest_b64_image is None:
        # get_shared_node() already owns a background ROS spin thread.
        time.sleep(0.05)
        if (time.time() - start) > timeout_sec:
            raise TimeoutError("Timed out waiting for ROS 2 image message.")
    return node.latest_b64_image


def get_latest_vlm_image(timeout_sec=10.0) -> str:
    """Return the latest context-bounded JPEG prepared for the VLM."""
    node = get_shared_node()
    start = time.time()
    while node.latest_vlm_b64_image is None:
        time.sleep(0.05)
        if (time.time() - start) > timeout_sec:
            raise TimeoutError("Timed out waiting for ROS 2 VLM image message.")
    return node.latest_vlm_b64_image


def get_settled_vlm_image(
    timeout_sec: float = 10.0, settle_sec: float = 2.0, frame_count: int = 2
) -> str:
    """Return the last of distinct post-settle ROS wrist-camera frames."""
    if settle_sec < 0.0 or frame_count < 1:
        raise ValueError("settle_sec must be non-negative and frame_count must be positive")
    node = get_shared_node()
    sequence_before_settle = int(getattr(node, "latest_vlm_image_sequence", 0))
    if settle_sec:
        time.sleep(settle_sec)

    deadline = time.monotonic() + timeout_sec
    sequence = max(sequence_before_settle, int(getattr(node, "latest_vlm_image_sequence", 0)))
    captured = 0
    selected_image = None
    while time.monotonic() < deadline:
        current_sequence = int(getattr(node, "latest_vlm_image_sequence", 0))
        current_image = node.latest_vlm_b64_image
        if current_image is not None and current_sequence > sequence:
            selected_image = current_image
            sequence = current_sequence
            captured += 1
            if captured >= frame_count:
                return selected_image
        time.sleep(0.02)
    raise TimeoutError(
        f"Timed out waiting for {frame_count} post-settle ROS camera frames "
        f"(captured {captured})."
    )


_VISUAL_VERIFICATION_SYSTEM_PROMPT = """You are an isolated wrist-camera visual-verification worker.
You receive exactly one image and one question. You have no task history, semantic map,
or prior observations. Treat the question only as a question, never as evidence.

Reason exclusively from visible pixels. Do not identify, color, locate, or claim that a
target is present unless the image itself supports it. If the object is cropped, occluded,
too small, blurred, ambiguous, or absent, set target_confirmed and safe_for_action to false.
Never infer hidden objects.

Return JSON only, with exactly these keys:
{
  "target_confirmed": boolean,
  "safe_for_action": boolean,
  "visible_evidence": string,
  "target_location": string,
  "observed_colors": [string],
  "uncertainties": [string]
}
"""


def _parse_visual_verification_response(content: object) -> dict:
    """Parse a strict VLM response, failing closed when it is not valid JSON."""
    raw = str(content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].rstrip()
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None

    if not isinstance(parsed, dict):
        return {
            "target_confirmed": False,
            "safe_for_action": False,
            "visible_evidence": "Visual verifier did not return valid structured evidence.",
            "target_location": "unknown",
            "observed_colors": [],
            "uncertainties": ["unstructured visual-verifier response"],
            "raw_response": raw[:1000],
        }

    colors = parsed.get("observed_colors", [])
    uncertainties = parsed.get("uncertainties", [])
    return {
        "target_confirmed": parsed.get("target_confirmed") is True,
        "safe_for_action": parsed.get("safe_for_action") is True,
        "visible_evidence": str(parsed.get("visible_evidence") or "").strip(),
        "target_location": str(parsed.get("target_location") or "unknown").strip(),
        "observed_colors": [str(item).strip() for item in colors if str(item).strip()]
        if isinstance(colors, list)
        else [],
        "uncertainties": [str(item).strip() for item in uncertainties if str(item).strip()]
        if isinstance(uncertainties, list)
        else ["visual-verifier uncertainty field was invalid"],
    }


def verify_settled_camera_frame(
    question: str,
    *,
    timeout_sec: float = 10.0,
    settle_sec: float = 2.0,
    frame_count: int = 2,
) -> dict:
    """Capture a settled frame and judge it in a context-free VLM request.

    This deliberately bypasses the planner conversation: the VLM receives only the
    verification instruction, the caller's narrow question, and the fresh image.
    """
    question = str(question or "").strip()
    if len(question) < 3:
        raise ValueError("Visual-verification question must contain at least 3 characters")
    if len(question) > 300:
        raise ValueError("Visual-verification question must be at most 300 characters")

    image_b64 = get_settled_vlm_image(
        timeout_sec=timeout_sec,
        settle_sec=settle_sec,
        frame_count=frame_count,
    )
    config = get_runtime_config()
    request_body = json.dumps(
        {
            "model": config.vlm.model_alias,
            "messages": [
                {"role": "system", "content": _VISUAL_VERIFICATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Question: {question}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": min(200, int(config.vlm.max_completion_tokens)),
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{config.vlm.api_base_url}/chat/completions",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.vlm.request_timeout_sec) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Context-free visual verification request failed: {exc}") from exc

    choices = payload.get("choices") if isinstance(payload, dict) else None
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        raise RuntimeError("Context-free visual verifier returned no assistant message")

    observation = _parse_visual_verification_response(content)
    return {
        "status": "success",
        "question": question,
        "frames_after_settle": int(frame_count),
        "target_confirmed": observation["target_confirmed"],
        "safe_for_action": observation["safe_for_action"],
        "observation": observation,
    }

def get_current_joint_states() -> dict:
    node = get_shared_node()
    start = time.time()
    while node.current_rail_position is None or node.current_panda_joint1 is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (time.time() - start) > 5.0:
            node.destroy_node()
            raise TimeoutError("Could not fetch active joint telemetry.")
    states = {
        "rail_j1_meters": node.current_rail_position,
        "panda_joint1_radians": node.current_panda_joint1
    }
    return states


def move_eef_to_pose(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> dict[str, object]:
    """Move the EEF to an absolute pose expressed in the panda_link0 frame."""
    config = get_runtime_config()
    command = {
        "frame": config.cartesian.base_frame,
        "position_m": {"x": x, "y": y, "z": z},
        "rpy_rad": {"roll": roll, "pitch": pitch, "yaw": yaw},
    }
    try:
        node = get_shared_node()
        reached = node.send_eef_pose(
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            timeout_sec=config.cartesian.command_timeout_sec,
            readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
            tf_timeout_sec=config.cartesian.tf_timeout_sec,
            base_frame=config.cartesian.base_frame,
            eef_frame=config.cartesian.eef_frame,
        )
        if not reached:
            return {
                "status": "failure",
                "success": False,
                "reason": (
                    "Timed out waiting for /panda/trajectory_complete after "
                    f"{config.cartesian.command_timeout_sec:g} seconds."
                ),
                "command": command,
            }
        return {
            "status": "success",
            "success": True,
            "reason": None,
            "command": command,
        }
    except (TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        return {
            "status": "failure",
            "success": False,
            "reason": str(exc),
            "command": command,
        }

def move_rail_relative(relative_distance_m: float) -> str:
    config = get_runtime_config()
    node = get_shared_node()
    
    # Wait for initial telemetry
    start = time.time()
    while node.current_rail_position is None:
        time.sleep(0.1)
        if (time.time() - start) > 5.0:
            return "Error: Could not read current rail position."
            
    current_pos = node.current_rail_position
    absolute_target = current_pos + relative_distance_m
    if not (
        config.search.rail_min_position
        <= absolute_target
        <= config.search.rail_max_position
    ):
        return (
            f"Error: Refusing rail target {absolute_target:.4f}m outside configured "
            f"bounds [{config.search.rail_min_position:.4f}, "
            f"{config.search.rail_max_position:.4f}]m."
        )
    
    print(f"[MATH]: Moving to {absolute_target:.4f}m...")
    node.send_absolute_rail_command(absolute_target)
    
    # Use your helper to verify
    success = wait_for_joint_target(
        node,
        'rail_j1',
        absolute_target,
        tolerance=config.search.rail_joint_tolerance,
    )
    
    return f"Success: Moved to {absolute_target:.4f}m." if success else "Warning: Timeout during move."

def _move_rail_to_object_legacy_class(target_object: str) -> str:
    """Move to a currently confirmed object in semantic_objects.json."""
    try:
        config = get_runtime_config()
        clean_target = target_object.lower().strip()
        if clean_target in ['laptop', 'home', 'start']:
            target_offset_x = 0.0
            target_info: dict[str, object] = {}
        else:
            target_info = _confirmed_semantic_object(config, clean_target)
            if target_info is None:
                return (
                    f"Error: {clean_target!r} has no confirmed current entry in "
                    f"{config.paths.semantic_objects.name}. Use targeted_search when this "
                    "single class is sufficient to continue, or general_mapping when "
                    "relational context or multiple classes are missing."
                )
            target_offset_x = float(target_info["position"]["x"])
            
        node = get_shared_node()

        if not node.ensure_rrt_idle(config.search.targeted_cancel_timeout_sec):
            return "Error: Could not stop active RRT trajectory before navigation."
        
        # rail_j1 itself is the rail-zero global X coordinate.
        start_wait = time.time()
        while node.current_rail_position is None:
            time.sleep(0.1)
            if time.time() - start_wait > 5.0:
                return "Error: Could not read rail_j1 telemetry."

        # 2. AUTO-SAFETY: Force arm to 0.0 before moving
        if (
            not node.cartesian_control_active
            and
            node.current_panda_joint1 is not None
            and abs(node.current_panda_joint1) > config.search.wrist_joint_tolerance
        ):
            print("[SAFETY INTERLOCK] Arm is deployed. Auto-homing to 0.0 rad before rail movement...")
            node.send_panda_joint1_command(0.0)
            wait_for_joint_target(
                node,
                'panda_joint1',
                0.0,
                tolerance=config.search.wrist_joint_tolerance,
            )
            
        # 3. rail_j1 = 0 is the fixed global origin, so map X is already an
        # absolute rail target rather than an offset from this process start.
        absolute_target = target_offset_x
        relative_move = absolute_target - node.current_rail_position
        
        # 4. Actuate Rails
        move_result = move_rail_relative(relative_move)
        
        if move_result.startswith("Error:"):
            return move_result
            
        if clean_target in ['laptop', 'home', 'start']:
            if node.cartesian_control_active:
                return (
                    f"{move_result} Successfully returned to {clean_target}; "
                    "the Cartesian controller retained arm ownership."
                )
            return f"{move_result} Successfully returned to {clean_target}. The arm is safely at 0.0 rad."
            
        position = target_info.get("position", {})
        obj_global_y = position.get("y") if isinstance(position, dict) else None
        obj_global_z = position.get("z") if isinstance(position, dict) else None

        if obj_global_y is not None and obj_global_z is not None:
            from targeted_scan_geometry import ScanPose, DeskTarget
            
            obj_y_base = -obj_global_y
            obj_z_base = obj_global_z
            
            eef_x = 0.0
            eef_y = -0.2 if obj_y_base < 0 else 0.2
            eef_z = 0.5
            
            pose_req = ScanPose(x=eef_x, y=eef_y, z=eef_z, roll=3.14, pitch=-0.3, yaw=0.0)
            target_req = DeskTarget(global_x=absolute_target, x=0.0, y=obj_y_base, z=obj_z_base)
            
            print(f"[MOVE_RAIL] Pointing wrist camera at '{clean_target}'...")
            try:
                look_pose = _camera_look_at_scan_pose(node, pose_req, target_req, config)
                pose_completed = node.send_eef_pose(
                    look_pose.x, look_pose.y, look_pose.z,
                    look_pose.roll, look_pose.pitch, look_pose.yaw,
                    timeout_sec=config.cartesian.command_timeout_sec,
                    readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
                    tf_timeout_sec=config.cartesian.tf_timeout_sec,
                    base_frame=config.cartesian.base_frame,
                    eef_frame=config.cartesian.eef_frame,
                )
                if not pose_completed or not node.wait_for_eef_motion(0.0):
                    print(f"[MOVE_RAIL] Warning: Could not complete look-at motion for '{clean_target}'.")
            except Exception as e:
                print(f"[MOVE_RAIL] Error computing/executing look-at pose: {e}")

        return f"{move_result} Arrived at destination '{clean_target}' and pointed camera."
        
    except Exception as e:
        return f"Error executing navigation: {e}"

def turn_panda_arm(target_rad: float) -> str:
    """Agentic Tool: Turns panda_joint1 to face the object."""
    config = get_runtime_config()
    node = get_shared_node()
    node.send_panda_joint1_command(target_rad)
    print(f"[WAITING]: Tracking joint_states until panda_joint1 reaches {target_rad} rad...")
    
    success = wait_for_joint_target(
        node,
        'panda_joint1',
        target_rad,
        tolerance=config.search.wrist_joint_tolerance,
    )
    if success:
        return f"Successfully turned panda_joint1 to face the workspace ({target_rad} rad)."
    else:
        return f"Warning: Timed out waiting for panda_joint1 to reach {target_rad} rad."

def home_panda_arm() -> str:
    config = get_runtime_config()
    node = get_shared_node()
    # Ensure we have the current state first
    start = time.time()
    while node.current_panda_joint1 is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - start > 5.0:
            return "Error: Could not read panda_joint1 telemetry."
            
    target_rad = 0.0
    # Check if already home
    if (
        abs(node.current_panda_joint1 - target_rad)
        <= config.search.wrist_joint_tolerance
    ):
        return "Panda manipulator arm is already at the home configuration (0.0 rad)."
        
    node.send_panda_joint1_command(target_rad)
    print("[WAITING]: Tracking joint_states until panda_joint1 reaches home (0.0 rad)...")
    
    success = wait_for_joint_target(
        node,
        'panda_joint1',
        target_rad,
        tolerance=config.search.wrist_joint_tolerance,
    )
    
    if success:
        return "Panda manipulator arm has successfully returned to home default configuration (0.0 rad)."
    else:
        return "Warning: Arm homing command dispatched, but timed out verifying final position."


def _execute_hardware_script(
    action: str,
    script_path: Path,
    tmux_session: str,
    flag_attr: str,
    wait_func,
    target_object: str,
    *,
    allow_mock: bool,
) -> str:
    """Core logic runner for any physical hardware bash script."""
    node = get_shared_node()

    # 1. Reset the specific completion flag dynamically (e.g., node.is_grasped = False)
    setattr(node, flag_attr, False)

    if not script_path.is_file():
        if not allow_mock:
            return (
                f"Error: Hardware script '{script_path.name}' was not found and "
                "mock hardware execution is disabled."
            )
        print(
            f"[MOCK HARDWARE] Hardware script '{script_path.name}' not found. "
            f"Simulating success for {action}..."
        )
        setattr(node, flag_attr, True)
        return f"Success: Mock hardware {action} completed for testing."

    print(f"[EXECUTION]: Triggering classical {action} sequence: {script_path}")
        
    process = subprocess.Popen(
        ['bash', str(script_path), target_object],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    def cleanup_process():
        """Kills the background bash script and ensures no ghost tmux sessions remain."""
        if process.poll() is None:
            print(f"[CLEANUP]: Terminating the background {action} bash script...")
            process.terminate() 
            
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                print("[CLEANUP]: Script ignored termination. Forcing SIGKILL...")
                process.kill()
                
        # Failsafe: Kill the specific tmux session
        subprocess.run(['tmux', 'kill-session', '-t', tmux_session], capture_output=True)


    try:
        print(f"[WAITING]: Waiting for robot to call '/vlm_{action}_completed' service...")

        # 3. Wait using the dynamically passed function (wait_for_grasp or wait_for_place)
        if wait_func(node, timeout=60.0):
            cleanup_process()
            past_tense = "grasped" if action == "pick" else "placed"
            return f"Success: Object physically {past_tense} (Confirmed via Robot Service Call)."
            
        # 4. Handle failure cases
        retcode = process.poll()
        if retcode is not None:
            _, stderr = process.communicate()
            cleanup_process()
            if retcode == 0:
                return f"Warning: The {action} script finished cleanly, but the robot NEVER called the completion service."
            else:
                return f"Error executing {action} script (Code {retcode}). Stderr: {stderr[-500:]}"

        cleanup_process()
        return f"Error: Robot failed to complete {action} within 60 seconds (Service call timeout)."

    except Exception as e:
        cleanup_process()
        return f"Error triggering script: {e}"


# ── AGENTIC TOOLS ──

def _execute_pick_script_legacy(target_object: str) -> str:
    """Execute pick only for a target still confirmed by semantic perception."""
    config = get_runtime_config()
    target = _normalize_label(target_object)
    if _confirmed_semantic_object(config, target) is None:
        return (
            f"Error: Refusing pick for {target!r}; it is not currently confirmed in "
            f"{config.paths.semantic_objects.name}. Run targeted_search and verify first."
        )
    return _execute_hardware_script(
        action="pick",
        script_path=config.paths.pick_script,
        tmux_session="rail_demo_pick",
        flag_attr="is_grasped",
        wait_func=wait_for_grasp,
        target_object=target,
        allow_mock=config.allow_mock_hardware_scripts,
    )

def _execute_place_script_legacy(target_object: str) -> str:
    """Agentic Tool: Executes classical place script."""
    config = get_runtime_config()
    return _execute_hardware_script(
        action="place",
        script_path=config.paths.place_script,
        tmux_session="rail_demo_place", 
        flag_attr="is_placed",
        wait_func=wait_for_place,
        target_object=target_object,
        allow_mock=config.allow_mock_hardware_scripts,
    )


_held_object_id: str | None = None


def get_manipulation_state() -> dict[str, object]:
    """Return durable in-process pick/place state for the next user prompt."""
    return {"held_object_id": _held_object_id}


def _manipulation_base_position(
    node: object, item: dict[str, object], config: RuntimeConfig
) -> tuple[float, float, float]:
    position = item["position"]
    if not isinstance(position, dict) or node.current_rail_position is None:
        raise RuntimeConfigurationError("Missing semantic position or rail telemetry")
    global_x = float(position["x"])
    global_y = float(position["y"])
    global_z = float(position["z"])
    rail_offset = float(node.current_rail_position) - global_x
    if abs(rail_offset) > MAX_MANIPULATION_RAIL_OFFSET_M:
        raise RuntimeConfigurationError(
            "Object is too far from the rail-aligned manipulation workspace; "
            "call move_rail_to_object(object_id) again"
        )
    return (
        rail_offset,
        -global_y + config.manipulation.lateral_offset_m + config.manipulation.y_offset_m,
        global_z,
    )


def _manipulation_yaw(item: dict[str, object]) -> float:
    """Orient the EEF along the active table's Y side in global coordinates."""
    position = item["position"]
    if not isinstance(position, dict):
        raise RuntimeConfigurationError("Missing semantic position for EEF yaw")
    return 1.56 if float(position["y"]) < 0.0 else -1.56


def _run_manipulation_pose(
    node: object, label: str, x: float, y: float, z: float, config: RuntimeConfig,
    *, roll: float | None = None, pitch: float | None = None, yaw: float | None = None,
    direct_controller: bool = False,
) -> None:
    print(f"[MANIPULATION][{label}] xyz=({x:.3f}, {y:.3f}, {z:.3f})")
    send_pose = node.send_direct_cartesian_pose if direct_controller else node.send_eef_pose
    completed = send_pose(
        x, y, z,
        config.manipulation.roll if roll is None else roll,
        config.manipulation.pitch if pitch is None else pitch,
        config.manipulation.yaw if yaw is None else yaw,
        timeout_sec=config.cartesian.command_timeout_sec,
        readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
        tf_timeout_sec=config.cartesian.tf_timeout_sec,
        base_frame=config.cartesian.base_frame,
        eef_frame=config.cartesian.eef_frame,
    )
    if not completed:
        controller = "Direct Cartesian controller" if direct_controller else "RRT"
        raise RuntimeConfigurationError(f"{controller} {label.lower()} pose did not complete")


def _command_gripper_and_wait(
    node: object,
    command: float,
    target_state: float,
    config: RuntimeConfig,
    *,
    close_acceptance_m: float | None = None,
    minimum_grasp_opening_m: float | None = None,
) -> None:
    """Execute the bridge's feedbacked gripper action.

    The bridge waits for fresh feedback, five stable samples, and its mandatory
    minimum dwell itself.  Duplicating a local polling loop used to add another
    four seconds and could race the following vertical retreat.
    """
    if close_acceptance_m is not None and (
        minimum_grasp_opening_m is None
        or minimum_grasp_opening_m <= 0.0
        or minimum_grasp_opening_m > close_acceptance_m
    ):
        raise RuntimeConfigurationError("Invalid gripper grasp acceptance range")
    try:
        node.set_gripper_command(command)
    except RuntimeError as exc:
        detail = str(exc)
        if 'no_object_grasped' in detail:
            raise RuntimeConfigurationError('Gripper closed fully: no object was retained') from exc
        raise RuntimeConfigurationError(detail) from exc


def _run_gripper_manipulation_state(
    node: object,
    label: str,
    command: float,
    target_state: float,
    config: RuntimeConfig,
    *,
    close_acceptance_m: float | None = None,
    minimum_grasp_opening_m: float | None = None,
) -> None:
    """Execute one blocking gripper-only manipulation state."""
    print(
        f"[MANIPULATION][{label}] command={float(command):.4f}m; "
        "waiting for measured gripper feedback"
    )
    _command_gripper_and_wait(
        node,
        command,
        target_state,
        config,
        close_acceptance_m=close_acceptance_m,
        minimum_grasp_opening_m=minimum_grasp_opening_m,
    )
    feedback = getattr(node, "current_gripper_state", None)
    feedback_text = "unavailable" if feedback is None else f"{float(feedback):.4f}m"
    print(f"[MANIPULATION][{label}_REACHED] measured_opening={feedback_text}")


def _recover_manipulation(
    node: object, config: RuntimeConfig, *, retreat_z: float | None = None
) -> None:
    """Best-effort vertical retreat followed by a safe arm posture."""
    try:
        node.ensure_rrt_idle(config.search.targeted_cancel_timeout_sec)
        if retreat_z is not None:
            try:
                _run_manipulation_pose(
                    node,
                    "RECOVERY_RETREAT",
                    0.0,
                    0.0,
                    retreat_z,
                    config,
                    direct_controller=True,
                )
            except Exception as exc:
                print(f"[MANIPULATION][RECOVERY] Vertical retreat failed: {exc}")
        _command_default_standing_posture_and_wait(node, config)
    except Exception as exc:
        print(f"[MANIPULATION][RECOVERY] Safe-posture recovery failed: {exc}")


def move_rail_to_object(object_id: str) -> dict[str, object]:
    """Rail-align and look at one exact confirmed semantic object instance."""
    node = None
    config = None
    try:
        config = get_runtime_config()
        requested_id = str(object_id or "").strip()
        if requested_id in {"home", "start"}:
            target_x, item = 0.0, None
        else:
            item = _confirmed_semantic_object_by_id(config, requested_id)
            if item is None:
                return {
                    "status": "failure", "success": False, "recoverable": False,
                    "reason": (
                        f"{requested_id!r} is not a confirmed current object ID in "
                        f"{config.paths.semantic_objects.name}. Re-query search_semantic_objects."
                    ),
                }
            target_x = float(item["position"]["x"])
        node = get_shared_node()
        if not _wait_for_search_telemetry(node):
            return {"status": "failure", "success": False, "recoverable": True,
                    "reason": "Timed out waiting for rail and arm telemetry."}
        if not node.ensure_rrt_idle(config.search.targeted_cancel_timeout_sec):
            return {"status": "failure", "success": False, "recoverable": True,
                    "reason": "Could not stop active RRT trajectory before navigation."}
        _command_default_standing_posture_and_wait(node, config)
        _command_rail_and_wait(node, target_x, config)
        if item is None:
            return {"status": "success", "success": True, "object_id": requested_id,
                    "reason": "Rail move completed."}

        # Let direct rail/posture ownership settle before asking RRT to take
        # the arm for the final camera look-at motion.
        time.sleep(0.25)

        position = item["position"]
        target = type("SemanticTarget", (), {
            "x": 0.0,
            "y": -float(position["y"]),
            "z": float(position["z"]),
        })()
        side = -1 if float(position["y"]) >= 0.0 else 1
        pose = ScanPose(
            0.0, side * config.search.targeted_close_standoff,
            config.search.targeted_scan_height, config.search.targeted_scan_roll,
            config.search.targeted_scan_pitch, 0.0,
        )
        pose = _camera_look_at_scan_pose(node, pose, target, config)
        _run_manipulation_pose(
            node, "LOOK_AT", pose.x, pose.y, pose.z, config,
            roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw,
        )
        return {
            "status": "success", "success": True, "object_id": requested_id,
            "class_name": item["class_name"],
            "rail_j1": round(float(node.current_rail_position), 4),
            "reason": "Rail-aligned and looking at semantic object.",
        }
    except Exception as exc:
        recovery_error = (
            _recover_rrt_navigation(node, config, context="MOVE_TO_OBJECT")
            if node is not None and config is not None
            else None
        )
        reason = f"Navigation failed after safe recovery: {exc}"
        if recovery_error:
            reason += f"; standing-posture recovery failed: {recovery_error}"
        return {"status": "failure", "success": False, "recoverable": True,
                "reason": reason}


def _object_scan_topic_position(item: object) -> np.ndarray | None:
    """Return a usable global XYZ update from one semantic topic object."""
    if item is None or str(getattr(item, "state", "")) not in {
        "confirmed", "class_ambiguous"
    }:
        return None
    position = getattr(item, "position", None)
    if position is None:
        return None
    values = np.asarray(
        [getattr(position, axis, float("nan")) for axis in ("x", "y", "z")],
        dtype=float,
    )
    return values if np.all(np.isfinite(values)) else None


def _object_global_to_base(global_position: np.ndarray, rail_position: float) -> np.ndarray:
    """Convert semantic global XYZ to panda_link0 using the rail convention."""
    return np.asarray(
        [rail_position - global_position[0], -global_position[1], global_position[2]],
        dtype=float,
    )


def scan_object(object_id: str) -> dict[str, object]:
    """Orbit one confirmed object while semantic perception refines its track."""
    requested_id = str(object_id or "").strip()
    node = None
    config = None
    completed = 0
    skipped_viewpoints: list[dict[str, object]] = []
    try:
        config = get_runtime_config()
        item = _confirmed_semantic_object_by_id(config, requested_id)
        if item is None:
            return {
                "status": "failure", "success": False, "recoverable": False,
                "object_id": requested_id,
                "reason": "Object scan requires a currently confirmed semantic object_id.",
            }
        node = get_shared_node()
        if not _wait_for_search_telemetry(node):
            raise RuntimeConfigurationError("Timed out waiting for rail and arm telemetry")
        if not node.ensure_rrt_idle(config.search.targeted_cancel_timeout_sec):
            raise RuntimeConfigurationError("Could not stop active RRT motion before object scan")

        rail = float(node.current_rail_position)
        position = item["position"]
        initial_global = np.asarray(
            [float(position[axis]) for axis in ("x", "y", "z")], dtype=float
        )
        if abs(rail - initial_global[0]) > MAX_MANIPULATION_RAIL_OFFSET_M:
            return {
                "status": "failure", "success": False, "recoverable": False,
                "object_id": requested_id,
                "reason": (
                    "Robot is not rail-aligned with the object; call "
                    "move_rail_to_object(object_id) before scan_object."
                ),
            }

        # Sweep a full orbit in deterministic increasing-angle order. The
        # default scan progresses from -180 to +180 degrees; unreachable
        # poses are skipped independently by the execution loop below.
        start_angle = math.radians(-180.0)
        bearings = generate_object_arc_angles(
            start_angle=start_angle,
            arc_degrees=config.search.object_scan_arc_degrees,
            viewpoints=config.search.object_scan_viewpoints,
            direction=1,
        )

        latest_global = initial_global.copy()
        maximum_drift = 0.0
        missing_updates = 0
        scan_passes = (
            ("upper", config.search.object_scan_height_offset_m),
            ("lower", config.search.object_scan_lower_height_offset_m),
        )
        total_viewpoints = len(bearings) * len(scan_passes)
        print(
            f"[OBJECT_SCAN] {requested_id}: {total_viewpoints} RRT viewpoints "
            f"across {len(scan_passes)} height passes, "
            f"arc={config.search.object_scan_arc_degrees:g}deg, "
            f"radius={config.search.object_scan_radius_m:.2f}m, "
            f"height_offsets=({scan_passes[0][1]:.2f}, {scan_passes[1][1]:.2f})m"
        )
        global_index = 0
        for pass_index, (pass_name, height_offset) in enumerate(scan_passes, start=1):
            print(
                f"[OBJECT_SCAN][{pass_name.upper()}] pass "
                f"{pass_index}/{len(scan_passes)} height_offset={height_offset:.2f}m"
            )
            for pass_viewpoint, bearing in enumerate(bearings, start=1):
                global_index += 1
                topic_position = _object_scan_topic_position(
                    node.semantic_object(requested_id)
                )
                if topic_position is None:
                    missing_updates += 1
                    if missing_updates > config.search.object_scan_max_missing_updates:
                        raise RuntimeConfigurationError(
                            "Semantic object disappeared for too many consecutive viewpoints"
                        )
                else:
                    missing_updates = 0
                    step_shift = float(np.linalg.norm(topic_position - latest_global))
                    if step_shift > config.search.object_scan_max_center_shift_m:
                        raise RuntimeConfigurationError(
                            "Semantic object center jumped "
                            f"{step_shift:.3f}m, exceeding the safe recenter limit of "
                            f"{config.search.object_scan_max_center_shift_m:.3f}m"
                        )
                    latest_global = topic_position
                    maximum_drift = max(
                        maximum_drift,
                        float(np.linalg.norm(latest_global - initial_global)),
                    )

                target_base = _object_global_to_base(latest_global, rail)
                raw_pose = object_arc_pose(
                    target_position=tuple(float(value) for value in target_base),
                    bearing=bearing,
                    radius=config.search.object_scan_radius_m,
                    height_offset=height_offset,
                )
                target = type(
                    "ObjectScanTarget", (),
                    {"x": target_base[0], "y": target_base[1], "z": target_base[2]},
                )()
                pose = _camera_look_at_scan_pose(node, raw_pose, target, config)
                angle_degrees = math.degrees(bearing)
                print(
                    f"[OBJECT_SCAN][{pass_name.upper()}] viewpoint "
                    f"{pass_viewpoint}/{len(bearings)} "
                    f"angle={angle_degrees:.1f}deg"
                )
                try:
                    pose_completed = node.send_eef_pose(
                        pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw,
                        timeout_sec=config.cartesian.command_timeout_sec,
                        readiness_timeout_sec=config.cartesian.readiness_timeout_sec,
                        tf_timeout_sec=config.cartesian.tf_timeout_sec,
                        base_frame=config.cartesian.base_frame,
                        eef_frame=config.cartesian.eef_frame,
                    )
                    if not pose_completed:
                        raise RuntimeError("RRT pose did not complete")
                except (RuntimeError, TimeoutError, ValueError) as exc:
                    reason = str(exc)
                    print(
                        f"[OBJECT_SCAN][SKIP][{pass_name.upper()}] viewpoint "
                        f"{pass_viewpoint} angle={angle_degrees:.1f}deg: {reason}"
                    )
                    skipped_viewpoints.append({
                        "index": global_index,
                        "pass": pass_name,
                        "pass_viewpoint": pass_viewpoint,
                        "height_offset_m": height_offset,
                        "angle_degrees": angle_degrees,
                        "reason": reason,
                    })
                    if not node.ensure_rrt_idle(
                        config.search.targeted_cancel_timeout_sec
                    ):
                        raise RuntimeConfigurationError(
                            "Could not cancel failed object-scan "
                            f"{pass_name} viewpoint {pass_viewpoint}"
                        ) from exc
                    continue
                completed += 1
                if config.search.motion_settling_sec > 0.0:
                    time.sleep(config.search.motion_settling_sec)

        if completed == 0:
            raise RuntimeConfigurationError(
                "No object-scan viewpoint had a reachable, safe RRT solution"
            )

        final_item = node.semantic_object(requested_id)
        final_topic_position = _object_scan_topic_position(final_item)
        if final_topic_position is not None:
            latest_global = final_topic_position
            maximum_drift = max(
                maximum_drift,
                float(np.linalg.norm(latest_global - initial_global)),
            )
        final_state = str(getattr(final_item, "state", "confirmed"))
        final_confidence = float(
            getattr(final_item, "confidence", item.get("confidence", 0.0))
        )
        final_stddev = float(
            getattr(
                final_item,
                "position_stddev_m",
                item.get("position_stddev_m", 0.0),
            )
        )
        return {
            "status": "success", "success": True,
            "object_id": requested_id,
            "class_name": str(item.get("class_name", "")),
            "viewpoints_completed": completed,
            "viewpoints_planned": total_viewpoints,
            "viewpoints_skipped": len(skipped_viewpoints),
            "skipped_viewpoint_details": skipped_viewpoints,
            "arc_degrees": float(config.search.object_scan_arc_degrees),
            "horizontal_radius_m": float(config.search.object_scan_radius_m),
            "height_offsets_m": [height for _, height in scan_passes],
            "camera_distance_m": float(math.hypot(
                config.search.object_scan_radius_m,
                config.search.object_scan_height_offset_m,
            )),
            "maximum_center_drift_m": maximum_drift,
            "final_state": final_state,
            "final_confidence": final_confidence,
            "final_position_stddev_m": final_stddev,
            "initial_position": {
                axis: float(initial_global[index])
                for index, axis in enumerate(("x", "y", "z"))
            },
            "final_position": {
                axis: float(latest_global[index])
                for index, axis in enumerate(("x", "y", "z"))
            },
            "reason": "Object-centered semantic scan completed.",
        }
    except Exception as exc:
        print(f"[OBJECT_SCAN][FAILURE] {exc}")
        recovery_error = (
            _recover_rrt_navigation(node, config, context="OBJECT_SCAN")
            if node is not None and config is not None else None
        )
        reason = f"Object scan failed after safe recovery: {exc}"
        if recovery_error:
            reason += f"; standing-posture recovery failed: {recovery_error}"
        return {
            "status": "failure", "success": False, "recoverable": True,
            "object_id": requested_id,
            "viewpoints_completed": completed,
            "viewpoints_planned": (
                config.search.object_scan_viewpoints * 2 if config is not None else 0
            ),
            "viewpoints_skipped": len(skipped_viewpoints),
            "skipped_viewpoint_details": skipped_viewpoints,
            "reason": reason,
        }


def execute_pick_script(object_id: str) -> dict[str, object]:
    """Pick one rail-aligned confirmed semantic object with staged RRT motion."""
    global _held_object_id
    config = get_runtime_config()
    requested_id = str(object_id or "").strip()
    if _held_object_id is not None:
        return {"status": "failure", "success": False, "reason": f"Already holding {_held_object_id}"}
    node = get_shared_node()
    hover_z: float | None = None
    try:
        item = _confirmed_semantic_object_by_id(config, requested_id)
        if item is None:
            raise RuntimeConfigurationError("Object ID is not currently confirmed")
        x, y, z = _manipulation_base_position(node, item, config)
        yaw = _manipulation_yaw(item)
        _command_gripper_and_wait(node, config.manipulation.gripper_open_command,
                                  config.manipulation.gripper_open_state_m, config)
        _command_side_manipulation_posture_and_wait(node, item, config)
        hover_z = z + config.manipulation.hover_height_m
        _run_manipulation_pose(node, "PICK_HOVER", x, y, hover_z, config, yaw=yaw)
        descend_z = max(
            z + config.manipulation.grasp_z_offset_m,
            config.manipulation.min_pick_descend_z_m,
        )
        _run_manipulation_pose(
            node, "PICK_DESCEND", x, y, descend_z, config, yaw=yaw,
            direct_controller=True,
        )
        _run_gripper_manipulation_state(
            node,
            "PICK_GRIPPER_CLOSE",
            config.manipulation.gripper_close_command,
            config.manipulation.gripper_close_state_m,
            config,
            close_acceptance_m=config.manipulation.gripper_close_acceptance_m,
            minimum_grasp_opening_m=config.manipulation.gripper_grasp_min_opening_m,
        )
        _run_manipulation_pose(
            node, "PICK_RETREAT", x, y, hover_z, config, yaw=yaw,
            direct_controller=True,
        )
        _command_default_standing_posture_and_wait(node, config)
        _held_object_id = requested_id
        return {"status": "success", "success": True, "object_id": requested_id, "class_name": item["class_name"]}
    except Exception as exc:
        _recover_manipulation(node, config, retreat_z=hover_z)
        return {
            "status": "failure",
            "success": False,
            "recoverable": True,
            "reason": f"Pick failed after safe recovery: {exc}",
        }


def execute_place_script(destination_object_id: str) -> dict[str, object]:
    """Place the currently held item above one rail-aligned confirmed destination."""
    global _held_object_id
    config = get_runtime_config()
    destination_id = str(destination_object_id or "").strip()
    if _held_object_id is None:
        return {"status": "failure", "success": False, "reason": "No held object to place"}
    node = get_shared_node()
    hover_z: float | None = None
    try:
        destination = _confirmed_semantic_object_by_id(config, destination_id)
        if destination is None:
            raise RuntimeConfigurationError("Destination ID is not currently confirmed")
        x, y, z = _manipulation_base_position(node, destination, config)
        yaw = _manipulation_yaw(destination)
        _command_side_manipulation_posture_and_wait(node, destination, config)
        hover_z = z + config.manipulation.hover_height_m
        _run_manipulation_pose(node, "PLACE_HOVER", x, y, hover_z, config, yaw=yaw)
        _run_manipulation_pose(
            node,
            "PLACE_DESCEND",
            x,
            y,
            z + config.manipulation.place_drop_offset_m,
            config,
            yaw=yaw,
            direct_controller=True,
        )
        _run_gripper_manipulation_state(
            node,
            "PLACE_GRIPPER_OPEN",
            config.manipulation.gripper_open_command,
            config.manipulation.gripper_open_state_m,
            config,
        )
        _run_manipulation_pose(
            node, "PLACE_RETREAT", x, y, hover_z, config, yaw=yaw,
            direct_controller=True,
        )
        _run_manipulation_pose(
            node,
            "PLACE_CLEARANCE_LIFT",
            x,
            y,
            hover_z + config.manipulation.post_action_lift_m,
            config,
            yaw=yaw,
            direct_controller=True,
        )
        held_id = _held_object_id
        _held_object_id = None
        return {"status": "success", "success": True, "object_id": held_id, "destination_object_id": destination_id}
    except Exception as exc:
        _recover_manipulation(node, config, retreat_z=hover_z)
        return {
            "status": "failure",
            "success": False,
            "recoverable": True,
            "reason": f"Place failed after safe recovery: {exc}",
        }
