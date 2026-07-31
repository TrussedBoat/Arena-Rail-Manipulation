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


VLM_WARMUP_LATENCY_LIMIT_SEC = 5.0
MAX_CENTERING_ITERATIONS = 50
MAX_REACQUISITION_FRAMES = 2
MAX_CENTERED_CONFIDENCE_RECHECKS = 2
SEARCH_TELEMETRY_TIMEOUT_SEC = 10.0
JOINT_MOTION_TIMEOUT_SEC = 50.0
JOINT_POLL_INTERVAL_SEC = 0.05
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
                    
                    if label == normalized_target or confidence > 0.3:
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
                    
        cv2.imshow("YOLO Search", frame)
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
    """Agentic Tool: Initializes the global ROS 2 joint interface node."""
    try:
        node = get_shared_node()
        return "Success: Global joint controller / ROS 2 interface is active and ready to receive movement commands."
    except Exception as e:
        return f"Error starting controller: {e}"


def _failure_result(
    target: str,
    reason: str,
    *,
    state: str,
    detection: Mapping[str, object] | None = None,
    observations: int = 0,
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
    }


def _wait_for_search_telemetry(node: object) -> bool:
    deadline = time.monotonic() + SEARCH_TELEMETRY_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if (
            getattr(node, "current_rail_position", None) is not None
            and getattr(node, "initial_rail_position", None) is not None
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


def _command_rail_and_wait(node: object, target: float, config: RuntimeConfig) -> None:
    if not config.search.rail_min_position <= target <= config.search.rail_max_position:
        raise RuntimeConfigurationError(
            f"Refusing rail target {target:.4f}m outside configured bounds"
        )
    starting_position = getattr(node, "current_rail_position", None)
    node.send_absolute_rail_command(float(target))
    if not _wait_for_commanded_joint(
        node,
        "rail_j1",
        target,
        config.search.rail_joint_tolerance,
        starting_position=starting_position,
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


def _wait_for_commanded_joint(
    node: object,
    joint_name: str,
    target: float,
    tolerance: float,
    *,
    starting_position: object,
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
    deadline = time.monotonic() + JOINT_MOTION_TIMEOUT_SEC

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
            if at_target and made_progress:
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


def _recover_lost_target(
    node: object,
    config: RuntimeConfig,
    target: str,
    *,
    last_seen_rail: float,
) -> tuple[dict[str, object] | None, int]:
    current_rail = float(node.current_rail_position)
    if abs(current_rail - last_seen_rail) > config.search.rail_joint_tolerance:
        print(
            "[SEARCH][RECOVER] Backtracking rail from "
            f"{current_rail:.4f}m to last observation at {last_seen_rail:.4f}m."
        )
        _command_rail_and_wait(node, last_seen_rail, config)

    for attempt in range(1, MAX_REACQUISITION_FRAMES + 1):
        print(
            f"[SEARCH][RECOVER] Reacquisition frame {attempt}/"
            f"{MAX_REACQUISITION_FRAMES}."
        )
        detection = _capture_stationary_detection(
            node,
            config,
            target,
            expected_rail=last_seen_rail,
            expected_wrist=config.search.final_centering_angle,
        )
        if detection["status"] != "not_found":
            return detection, attempt
    return None, MAX_REACQUISITION_FRAMES


def _center_target_on_rail(
    node: object,
    config: RuntimeConfig,
    target: str,
    trigger_detection: Mapping[str, object],
    found_wrist_angle: float,
) -> dict[str, object]:
    print("[SEARCH][CENTER] Moving wrist to the calibrated centering angle.")
    _command_wrist_and_wait(
        node, config.search.final_centering_angle, config
    )
    expected_rail = float(node.current_rail_position)
    last_seen_rail = expected_rail
    last_detection: Mapping[str, object] = trigger_detection
    observation = _capture_stationary_detection(
        node,
        config,
        target,
        expected_rail=expected_rail,
        expected_wrist=config.search.final_centering_angle,
    )
    observations = 1
    centered_confidence_rechecks = 0

    for iteration in range(1, MAX_CENTERING_ITERATIONS + 1):
        if observation["status"] == "not_found":
            recovered, recovery_observations = _recover_lost_target(
                node,
                config,
                target,
                last_seen_rail=last_seen_rail,
            )
            observations += recovery_observations
            if recovered is None:
                return {
                    "status": "failure",
                    "reason": "Target was lost during centering and could not be reacquired.",
                    "detection": last_detection,
                    "iterations": iteration,
                    "observations": observations,
                }
            observation = recovered
            expected_rail = last_seen_rail

        current_rail = float(node.current_rail_position)
        last_seen_rail = current_rail
        last_detection = observation
        horizontal_error = float(observation["horizontal_error"])
        print(
            f"[SEARCH][CENTER] iteration={iteration}, rail={current_rail:.4f}m, "
            f"error={horizontal_error:.4f}, confidence="
            f"{float(observation['confidence']):.4f}."
        )

        if abs(horizontal_error) <= config.search.horizontal_center_tolerance:
            if bool(observation["final_eligible"]):
                return {
                    "status": "success",
                    "reason": None,
                    "detection": observation,
                    "iterations": iteration,
                    "observations": observations,
                }
            centered_confidence_rechecks += 1
            if centered_confidence_rechecks > MAX_CENTERED_CONFIDENCE_RECHECKS:
                return {
                    "status": "failure",
                    "reason": (
                        "Target was centered but confidence did not become strictly "
                        f"greater than {config.yolo.confidence_threshold:.2f}."
                    ),
                    "detection": observation,
                    "iterations": iteration,
                    "observations": observations,
                }
            observation = _capture_stationary_detection(
                node,
                config,
                target,
                expected_rail=current_rail,
                expected_wrist=config.search.final_centering_angle,
            )
            observations += 1
            continue

        centered_confidence_rechecks = 0
        raw_correction = config.search.centering_gain * horizontal_error
        correction_magnitude = min(
            config.search.centering_max_step,
            max(config.search.centering_min_step, abs(raw_correction)),
        )
        correction = math.copysign(correction_magnitude, horizontal_error)
        next_rail = min(
            config.search.rail_max_position,
            max(config.search.rail_min_position, current_rail + correction),
        )
        if abs(next_rail - current_rail) <= 1e-12:
            return {
                "status": "failure",
                "reason": "Centering correction is blocked by a configured rail limit.",
                "detection": observation,
                "iterations": iteration,
                "observations": observations,
            }

        _command_rail_and_wait(node, next_rail, config)
        expected_rail = next_rail
        observation = _capture_stationary_detection(
            node,
            config,
            target,
            expected_rail=expected_rail,
            expected_wrist=config.search.final_centering_angle,
        )
        observations += 1

    return {
        "status": "failure",
        "reason": (
            f"Centering did not converge within {MAX_CENTERING_ITERATIONS} iterations."
        ),
        "detection": last_detection,
        "iterations": MAX_CENTERING_ITERATIONS,
        "observations": observations,
    }


def _persist_dynamic_coordinate(path: Path, target: str, x: float) -> None:
    if not math.isfinite(x):
        raise RuntimeConfigurationError("Refusing to persist a non-finite coordinate")
    if not path.parent.is_dir():
        raise RuntimeConfigurationError(
            f"Dynamic-coordinate parent directory does not exist: {path.parent}"
        )

    coordinates: dict[str, object] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as coordinate_file:
                existing_coordinates = json.load(coordinate_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeConfigurationError(
                f"Could not read dynamic-coordinate JSON {path}: {exc}"
            ) from exc
        if not isinstance(existing_coordinates, dict):
            raise RuntimeConfigurationError(
                f"Dynamic-coordinate JSON must contain an object: {path}"
            )
        coordinates.update(existing_coordinates)

    coordinates[target] = {"x": float(x)}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(coordinates, temporary_file, indent=4, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeConfigurationError(
            f"Could not persist dynamic coordinate to {path}: {exc}"
        ) from exc


def search_and_locate_with_yolo(target_object: str) -> dict[str, object]:
    """Search, center, and persist one canonical target using stationary YOLO frames."""
    target = _normalize_label(target_object or "")
    if not target:
        return _failure_result(
            target,
            "A non-empty canonical target label is required.",
            state="initialize",
        )

    observations = 0
    last_centering_failure: Mapping[str, object] | None = None
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
                "Timed out waiting for rail, wrist, and session-origin telemetry.",
                state="initialize",
            )

        initial_rail_position = float(node.initial_rail_position)
        current_rail_position = float(node.current_rail_position)
        if not math.isfinite(initial_rail_position) or not math.isfinite(
            current_rail_position
        ):
            return _failure_result(
                target,
                "Rail telemetry or session origin is not finite.",
                state="initialize",
            )
        limits = [config.search.rail_min_position, config.search.rail_max_position]
        start_limit = limits[0] if abs(current_rail_position - limits[0]) < abs(current_rail_position - limits[1]) else limits[1]

        print(
            f"[SEARCH][INITIALIZE] target={target!r}, "
            f"origin={initial_rail_position:.4f}m."
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
            node.send_rail_and_joint6_command(
                rail_val=target_limit, 
                rail_speed=config.search.rail_speed, 
                j6_val=j6_target, 
                j6_speed=config.search.j6_speed
            )

            while True:
                # Handle j6 oscillation
                if node.current_panda_joint6 is not None:
                    if abs(node.current_panda_joint6 - j6_target) < 0.15:
                        j6_target = j6_limits[0] if j6_target == j6_limits[1] else j6_limits[1]
                        print(f"[SEARCH][J6] Flipped target to {j6_target:.2f}rad")
                        node.send_panda_joint6_command(j6_target, speed=config.search.j6_speed)
                        
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
                    "[SEARCH][DETECTED] Candidate found; halting sweep and "
                    "transitioning to centering."
                )
                if node.current_panda_joint6 is not None:
                    node.send_rail_and_joint6_command(
                        rail_val=node.current_rail_position,
                        rail_speed=0.0,
                        j6_val=node.current_panda_joint6,
                        j6_speed=0.0
                    )
                else:
                    node.send_absolute_rail_command(node.current_rail_position)
                
                centering = _center_target_on_rail(
                    node, config, target, detection, wrist_angle
                )
                observations += int(centering["observations"])
                if centering["status"] != "success":
                    last_centering_failure = centering
                    print(
                        f"[SEARCH][CENTER] {centering['reason']} Resuming continuous search sweep."
                    )
                    if hasattr(node, "send_panda_search_posture"):
                        node.send_panda_search_posture(wrist_angle, config.search.search_posture_j2, config.search.search_posture_j3, config.search.search_posture_j4, config.search.search_posture_j5, node.current_panda_joint6 if node.current_panda_joint6 else j6_limits[0], config.search.search_posture_j7)
                    else:
                        _command_wrist_and_wait(node, wrist_angle, config)
                    time.sleep(1.5)
                    node.send_rail_and_joint6_command(
                        rail_val=target_limit, 
                        rail_speed=config.search.rail_speed, 
                        j6_val=j6_target, 
                        j6_speed=config.search.j6_speed
                    )
                    continue

                final_detection = centering["detection"]
                target_centered = True
                
                current_absolute_rail = float(node.current_rail_position)
                object_x = current_absolute_rail - initial_rail_position
                try:
                    _persist_dynamic_coordinate(
                        config.paths.dynamic_semantic_coordinates,
                        target,
                        object_x,
                    )
                except RuntimeConfigurationError as exc:
                    return _failure_result(
                        target,
                        str(exc),
                        state="persist_coordinate",
                        detection=final_detection,
                        observations=observations,
                    )
                return {
                    "status": "success",
                    "success": True,
                    "state": "save_and_succeed",
                    "reason": None,
                    "target": target,
                    "confidence": final_detection["confidence"],
                    "bbox": final_detection["bbox"],
                    "center": final_detection["center"],
                    "horizontal_error": final_detection["horizontal_error"],
                    "x": object_x,
                    "absolute_rail_position": current_absolute_rail,
                    "initial_rail_position": initial_rail_position,
                    "centering_iterations": centering["iterations"],
                    "observations": observations,
                    "coordinate_path": str(
                        config.paths.dynamic_semantic_coordinates
                    ),
                }

        if not target_centered:
            last_detection = (
                last_centering_failure.get("detection")
                if last_centering_failure is not None
                else None
            )
            reason = "Full configured rail range was scanned without a valid centered detection."
            if last_centering_failure is not None:
                reason += f" Last centering failure: {last_centering_failure['reason']}"
            return _failure_result(
                target,
                reason,
                state="search_sweep",
                detection=last_detection,
                observations=observations,
            )

        last_detection = (
            last_centering_failure.get("detection")
            if last_centering_failure is not None
            else None
        )
        reason = "Full configured rail range was scanned without a valid centered detection."
        if last_centering_failure is not None:
            reason += f" Last centering failure: {last_centering_failure['reason']}"
        return _failure_result(
            target,
            reason,
            state="failure",
            detection=last_detection,
            observations=observations,
        )
    except Exception as exc:
        return _failure_result(
            target,
            f"Search aborted safely: {exc}",
            state="failure",
            observations=observations,
        )
        
def get_latest_ros_image(timeout_sec=10.0) -> str:
    node = get_shared_node()
    start = time.time()
    print("Waiting for image from ROS 2 topic...")
    while node.latest_b64_image is None:
        # get_shared_node() already owns a background ROS spin thread.
        time.sleep(0.05)
        if (time.time() - start) > timeout_sec:
            raise TimeoutError("Timed out waiting for ROS 2 image message.")
    return node.latest_b64_image

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

def move_rail_to_object(target_object: str) -> str:
    """Agentic Tool: Safely homes arm, calculates absolute world target, and moves rail."""
    try:
        config = get_runtime_config()
        distances: dict[str, dict[str, float]] = {}
        if config.paths.static_semantic_coordinates.is_file():
            try:
                with config.paths.static_semantic_coordinates.open("r", encoding="utf-8") as f:
                    distances.update(json.load(f))
            except (OSError, json.JSONDecodeError):
                pass
        if config.paths.dynamic_semantic_coordinates.is_file():
            try:
                with config.paths.dynamic_semantic_coordinates.open("r", encoding="utf-8") as f:
                    distances.update(json.load(f))
            except (OSError, json.JSONDecodeError):
                pass
            
        clean_target = target_object.lower().strip()
        
        # 1. Get Semantic Offsets
        if clean_target in ['laptop', 'home', 'start']:
            target_offset_x = 0.0
        elif clean_target in distances:
            target_offset_x = distances[clean_target]["x"]
        else:
            return f"Error: '{target_object}' not found in map."
            
        node = get_shared_node()
        
        # Wait for calibration
        start_wait = time.time()
        while node.current_rail_position is None or node.initial_rail_position is None:
            time.sleep(0.1)
            if time.time() - start_wait > 5.0:
                return "Error: Hardware not calibrated. Missing initial_rail_position."

        # 2. AUTO-SAFETY: Force arm to 0.0 before moving
        if (
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
            
        # 3. Hardware Origin Math
        # Absolute Target = Start Location (-1.09m) + Semantic Offset (2.31m)
        absolute_target = node.initial_rail_position + target_offset_x
        relative_move = absolute_target - node.current_rail_position
        
        # 4. Actuate Rails
        move_result = move_rail_relative(relative_move)
        
        if move_result.startswith("Error:"):
            return move_result
            
        if clean_target in ['laptop', 'home', 'start']:
            return f"{move_result} Successfully returned to {clean_target}. The arm is safely at 0.0 rad."
        return f"{move_result} Arrived at static destination '{clean_target}'."
        
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

def execute_pick_script(target_object: str) -> str:
    """Agentic Tool: Executes classical pick script."""
    config = get_runtime_config()
    return _execute_hardware_script(
        action="pick",
        script_path=config.paths.pick_script,
        tmux_session="rail_demo_pick",
        flag_attr="is_grasped",
        wait_func=wait_for_grasp,
        target_object=target_object,
        allow_mock=config.allow_mock_hardware_scripts,
    )

def execute_place_script(target_object: str) -> str:
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
