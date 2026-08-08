import json
import math
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ═══════════════════════════════════════════════════════════════════════════
# ── EDITABLE CONFIGURATION ──────────────────────────────────────────────
# Adjust these values to match your local machine. Everything else in this
# file is logic — you should not need to edit below this block.
# ═══════════════════════════════════════════════════════════════════════════

# --- Paths ---
DEFAULT_LLAMA_ROOT      = Path("/home/thinkstation-sim/workspace/llama.cpp")
DEFAULT_CONTROLLER_ROOT = Path("/home/thinkstation-sim/ros2_ws")

# --- VLM model filenames (relative to DEFAULT_LLAMA_ROOT/models/) ---
VLM_MODEL_FILENAME  = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
VLM_MMPROJ_FILENAME = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
VLM_MODEL_ALIAS     = "Qwen3VL-8B-Instruct-Q4_K_M"

# --- YOLO checkpoint ---
YOLO_CHECKPOINT_PATH: Path | None = PROJECT_ROOT / "agent_orchestrator/models/yolo26x.pt"


# --- VLM server settings ---
VLM_SERVER_HOST            = "127.0.0.1"
VLM_SERVER_PORT            = 8080
VLM_CONTEXT_SIZE           = 4096
VLM_PARALLEL_SLOTS         = 1
VLM_GPU_LAYERS             = 99
VLM_FLASH_ATTENTION        = True
VLM_MAX_COMPLETION_TOKENS  = 256
VLM_REQUEST_TIMEOUT_SEC    = 120.0
VLM_STARTUP_TIMEOUT_SEC    = 45.0
VLM_HEALTH_TIMEOUT_SEC     = 2.0
VLM_VRAM_BUDGET_GB         = 8.0
VLM_TOTAL_GPU_VRAM_GB      = 16.0
VLM_SERVER_EXTRA_ARGS      = ""

# --- YOLO settings ---
YOLO_CONFIDENCE_THRESHOLD  = 0.75
YOLO_IMAGE_SIZE            = 640
YOLO_DEVICE                = "cuda:0"
YOLO_MAX_DETECTIONS        = 100
YOLO_REQUIRED_CLASSES      = ("apple",)

# --- Search / rail parameters ---
SEARCH_RAIL_MIN             = -1.20  # metres
SEARCH_RAIL_MAX             =  1.60   # metres
SEARCH_RAIL_WAYPOINT_SPACING = 0.40  # metres between scan stops
# Override at runtime with YOLO_VLM_RAIL_SPEED (metres/second).
SEARCH_RAIL_SPEED           = 0.20
SEARCH_J6_SPEED             = 0.5
SEARCH_J6_MIN               = 0.8
SEARCH_J6_MAX               = 1.3

# --- Search Joint Angles ---
SEARCH_POSTURE_J2           = -0.7854
SEARCH_POSTURE_J3           = 1e-05
SEARCH_POSTURE_J4           = -1.54
SEARCH_POSTURE_J5           = 0.1
SEARCH_POSTURE_J7           = 0.7854
SEARCH_WRIST_ANGLES         = (-1.57, 0.0, 1.57)
SEARCH_MOTION_SETTLING_SEC  = 0.50
SEARCH_RAIL_TOLERANCE       = 0.02
SEARCH_WRIST_TOLERANCE      = 0.05

# --- Camera and reference frames ---
CAMERA_BASE_FRAME           = "panda_link0"
CAMERA_OPTICAL_FRAME        = "wrist_camera"
GLOBAL_ORIGIN_FRAME         = "global_origin"
CAMERA_FX                   = 907.00
CAMERA_FY                   = 905.69
CAMERA_CX                   = 567.74
CAMERA_CY                   = 488.32
DEPTH_PATCH_RADIUS          = 2

# --- RRT targeted-search scan ---
TARGETED_SCAN_VIEWPOINTS        = 7
TARGETED_CAPTURE_FPS            = 0.3 #lower it if the wrist camera is not able to capture fast enough
TARGETED_CANDIDATE_CONFIDENCE   = 0.50
TARGETED_DESK_WIDTH             = 1.0
TARGETED_TABLE_SCAN_Y           = 0.75
TARGETED_ARC_RADIUS             = 0.50
TARGETED_DESK_SURFACE_Z         = 0.0
TARGETED_SCAN_HEIGHT            = 0.50
TARGETED_SCAN_ROLL              = 3.14
TARGETED_SCAN_PITCH             = -0.3
TARGETED_CLOSE_STANDOFF         = 0.30
TARGETED_CANCEL_TIMEOUT_SEC     = 3.0

# --- Cartesian EEF settings ---
CARTESIAN_COMMAND_TIMEOUT_SEC   = 30.0
CARTESIAN_READY_TIMEOUT_SEC     = 5.0
CARTESIAN_TF_TIMEOUT_SEC        = 3.0
CARTESIAN_BASE_FRAME            = "panda_link0"
CARTESIAN_EEF_FRAME             = "eef"

# --- Runtime paths and behavior ---
ROS_SETUP_SCRIPT               = Path("/opt/ros/humble/setup.bash")
STATIC_COORDINATES_PATH        = PROJECT_ROOT / "semantic_distances.json"
DYNAMIC_COORDINATES_PATH       = PROJECT_ROOT / "semantic_distances_dynamic.json"
PICK_SCRIPT_RELATIVE_PATH      = Path("src/bringup/rail_demo_pick.sh")
PLACE_SCRIPT_RELATIVE_PATH     = Path("src/bringup/rail_demo_place.sh")
ALLOW_MOCK_HARDWARE_SCRIPTS    = True

# ═══════════════════════════════════════════════════════════════════════════


class RuntimeConfigurationError(RuntimeError):
    """Raised when runtime configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class VLMConfig:
    executable: Path
    model_path: Path
    mmproj_path: Path
    host: str
    port: int
    model_alias: str
    context_size: int
    parallel_slots: int
    gpu_layers: int
    flash_attention: bool
    max_completion_tokens: int
    request_timeout_sec: float
    startup_timeout_sec: float
    health_timeout_sec: float
    vram_budget_gb: float
    total_gpu_vram_gb: float
    extra_server_args: tuple[str, ...]

    @property
    def api_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"


@dataclass(frozen=True)
class YOLOConfig:
    checkpoint_path: Path
    confidence_threshold: float
    image_size: int
    device: str
    max_detections: int
    required_classes: tuple[str, ...]


@dataclass(frozen=True)
class SearchConfig:
    rail_min_position: float
    rail_max_position: float
    rail_waypoint_spacing: float
    rail_speed: float
    j6_speed: float
    j6_min: float
    j6_max: float
    search_posture_j2: float
    search_posture_j3: float
    search_posture_j4: float
    search_posture_j5: float
    search_posture_j7: float
    wrist_search_angles: tuple[float, ...]
    motion_settling_sec: float
    rail_joint_tolerance: float
    wrist_joint_tolerance: float
    # TF frame names
    camera_base_frame: str
    camera_optical_frame: str
    global_origin_frame: str
    # Hardcoded camera intrinsics (no camera_info topic available)
    camera_fx: float
    camera_fy: float
    camera_cx: float
    camera_cy: float
    depth_patch_radius: int  # px radius for median depth sampling
    targeted_scan_viewpoints: int
    targeted_capture_fps: float
    targeted_candidate_confidence: float
    targeted_desk_width: float
    targeted_table_scan_y: float
    targeted_arc_radius: float
    targeted_scan_height: float
    targeted_desk_surface_z: float
    targeted_scan_roll: float
    targeted_scan_pitch: float
    targeted_close_standoff: float
    targeted_cancel_timeout_sec: float


@dataclass(frozen=True)
class CartesianConfig:
    command_timeout_sec: float
    readiness_timeout_sec: float
    tf_timeout_sec: float
    base_frame: str
    eef_frame: str


@dataclass(frozen=True)
class PathConfig:
    static_semantic_coordinates: Path
    dynamic_semantic_coordinates: Path
    ros_setup_script: Path
    controller_workspace: Path
    controller_setup_script: Path
    pick_script: Path
    place_script: Path


@dataclass(frozen=True)
class RuntimeConfig:
    vlm: VLMConfig
    yolo: YOLOConfig
    search: SearchConfig
    cartesian: CartesianConfig
    paths: PathConfig
    allow_mock_hardware_scripts: bool


_runtime_config: RuntimeConfig | None = None


def _env_value(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default).strip()
    if not value:
        raise RuntimeConfigurationError(f"{name} must not be empty")
    return value


def _env_path(env: Mapping[str, str], name: str, default: Path) -> Path:
    raw_value = _env_value(env, name, str(default))
    return Path(os.path.expandvars(raw_value)).expanduser().resolve()


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = _env_value(env, name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeConfigurationError(f"{name} must be an integer, got {raw_value!r}") from exc


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw_value = _env_value(env, name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeConfigurationError(f"{name} must be a number, got {raw_value!r}") from exc
    if not math.isfinite(value):
        raise RuntimeConfigurationError(f"{name} must be finite, got {raw_value!r}")
    return value


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw_value = _env_value(env, name, "true" if default else "false").lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeConfigurationError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0; got {raw_value!r}"
    )


def _env_float_list(
    env: Mapping[str, str], name: str, default: Sequence[float]
) -> tuple[float, ...]:
    raw_value = _env_value(env, name, ",".join(str(value) for value in default))
    try:
        values = tuple(float(item.strip()) for item in raw_value.split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeConfigurationError(
            f"{name} must be a comma-separated list of numbers, got {raw_value!r}"
        ) from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeConfigurationError(f"{name} must contain one or more finite numbers")
    return values


def _env_string_list(
    env: Mapping[str, str], name: str, default: Sequence[str]
) -> tuple[str, ...]:
    raw_value = _env_value(env, name, ",".join(default))
    values = tuple(item.strip().lower() for item in raw_value.split(",") if item.strip())
    if not values:
        raise RuntimeConfigurationError(f"{name} must contain at least one value")
    return values


def load_runtime_config(env: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Load all runtime settings from environment variables without side effects."""
    source = os.environ if env is None else env
    controller_root = _env_path(
        source, "YOLO_VLM_CONTROLLER_WORKSPACE", DEFAULT_CONTROLLER_ROOT
    )
    llama_root = _env_path(source, "YOLO_VLM_LLAMA_ROOT", DEFAULT_LLAMA_ROOT)

    return RuntimeConfig(
        vlm=VLMConfig(
            executable=_env_path(
                source,
                "YOLO_VLM_LLAMA_EXECUTABLE",
                llama_root / "build/bin/llama-server",
            ),
            model_path=_env_path(
                source,
                "YOLO_VLM_MODEL_PATH",
                llama_root / "models" / VLM_MODEL_FILENAME,
            ),
            mmproj_path=_env_path(
                source,
                "YOLO_VLM_MMPROJ_PATH",
                llama_root / "models" / VLM_MMPROJ_FILENAME,
            ),
            host=_env_value(source, "YOLO_VLM_SERVER_HOST", VLM_SERVER_HOST),
            port=_env_int(source, "YOLO_VLM_SERVER_PORT", VLM_SERVER_PORT),
            model_alias=_env_value(
                source,
                "YOLO_VLM_MODEL_ALIAS",
                VLM_MODEL_ALIAS,
            ),
            context_size=_env_int(source, "YOLO_VLM_CONTEXT_SIZE", VLM_CONTEXT_SIZE),
            parallel_slots=_env_int(source, "YOLO_VLM_PARALLEL_SLOTS", VLM_PARALLEL_SLOTS),
            gpu_layers=_env_int(source, "YOLO_VLM_GPU_LAYERS", VLM_GPU_LAYERS),
            flash_attention=_env_bool(source, "YOLO_VLM_FLASH_ATTENTION", VLM_FLASH_ATTENTION),
            max_completion_tokens=_env_int(
                source, "YOLO_VLM_MAX_COMPLETION_TOKENS", VLM_MAX_COMPLETION_TOKENS
            ),
            request_timeout_sec=_env_float(
                source, "YOLO_VLM_REQUEST_TIMEOUT_SEC", VLM_REQUEST_TIMEOUT_SEC
            ),
            startup_timeout_sec=_env_float(
                source, "YOLO_VLM_STARTUP_TIMEOUT_SEC", VLM_STARTUP_TIMEOUT_SEC
            ),
            health_timeout_sec=_env_float(
                source, "YOLO_VLM_HEALTH_TIMEOUT_SEC", VLM_HEALTH_TIMEOUT_SEC
            ),
            vram_budget_gb=_env_float(source, "YOLO_VLM_VRAM_BUDGET_GB", VLM_VRAM_BUDGET_GB),
            total_gpu_vram_gb=_env_float(
                source, "YOLO_VLM_TOTAL_GPU_VRAM_GB", VLM_TOTAL_GPU_VRAM_GB
            ),
            extra_server_args=tuple(
                shlex.split(source.get("YOLO_VLM_SERVER_EXTRA_ARGS", VLM_SERVER_EXTRA_ARGS))
            ),
        ),
        yolo=YOLOConfig(
            checkpoint_path=_env_path(
                source,
                "YOLO_VLM_YOLO_CHECKPOINT",
                # YOLO_CHECKPOINT_PATH is None until weights are available.
                # Validation will skip file-existence checks when path is absent.
                YOLO_CHECKPOINT_PATH or PROJECT_ROOT / "agent_orchestrator/models/placeholder.pt",
            ),
            confidence_threshold=_env_float(
                source, "YOLO_VLM_YOLO_CONFIDENCE", YOLO_CONFIDENCE_THRESHOLD
            ),
            image_size=_env_int(source, "YOLO_VLM_YOLO_IMAGE_SIZE", YOLO_IMAGE_SIZE),
            device=_env_value(source, "YOLO_VLM_YOLO_DEVICE", YOLO_DEVICE),
            max_detections=_env_int(
                source, "YOLO_VLM_YOLO_MAX_DETECTIONS", YOLO_MAX_DETECTIONS
            ),
            required_classes=_env_string_list(
                source, "YOLO_VLM_YOLO_REQUIRED_CLASSES", YOLO_REQUIRED_CLASSES
            ),
        ),
        search=SearchConfig(
            rail_min_position=_env_float(source, "YOLO_VLM_RAIL_MIN", SEARCH_RAIL_MIN),
            rail_max_position=_env_float(source, "YOLO_VLM_RAIL_MAX", SEARCH_RAIL_MAX),
            rail_waypoint_spacing=_env_float(
                source, "YOLO_VLM_RAIL_WAYPOINT_SPACING", SEARCH_RAIL_WAYPOINT_SPACING
            ),
            rail_speed=_env_float(source, "YOLO_VLM_RAIL_SPEED", SEARCH_RAIL_SPEED),
            j6_speed=_env_float(source, "YOLO_VLM_J6_SPEED", SEARCH_J6_SPEED),
            j6_min=_env_float(source, "YOLO_VLM_J6_MIN", SEARCH_J6_MIN),
            j6_max=_env_float(source, "YOLO_VLM_J6_MAX", SEARCH_J6_MAX),
            search_posture_j2=_env_float(source, "YOLO_VLM_SEARCH_POSTURE_J2", SEARCH_POSTURE_J2),
            search_posture_j3=_env_float(source, "YOLO_VLM_SEARCH_POSTURE_J3", SEARCH_POSTURE_J3),
            search_posture_j4=_env_float(source, "YOLO_VLM_SEARCH_POSTURE_J4", SEARCH_POSTURE_J4),
            search_posture_j5=_env_float(source, "YOLO_VLM_SEARCH_POSTURE_J5", SEARCH_POSTURE_J5),
            search_posture_j7=_env_float(source, "YOLO_VLM_SEARCH_POSTURE_J7", SEARCH_POSTURE_J7),
            wrist_search_angles=_env_float_list(
                source, "YOLO_VLM_WRIST_SEARCH_ANGLES", SEARCH_WRIST_ANGLES
            ),
            motion_settling_sec=_env_float(
                source, "YOLO_VLM_MOTION_SETTLING_SEC", SEARCH_MOTION_SETTLING_SEC
            ),
            rail_joint_tolerance=_env_float(
                source, "YOLO_VLM_RAIL_JOINT_TOLERANCE", SEARCH_RAIL_TOLERANCE
            ),
            wrist_joint_tolerance=_env_float(
                source, "YOLO_VLM_WRIST_JOINT_TOLERANCE", SEARCH_WRIST_TOLERANCE
            ),
            # TF frame names published by Isaac Sim
            camera_base_frame=_env_value(source, "YOLO_VLM_CAMERA_BASE_FRAME", CAMERA_BASE_FRAME),
            camera_optical_frame=_env_value(source, "YOLO_VLM_CAMERA_OPTICAL_FRAME", CAMERA_OPTICAL_FRAME),
            global_origin_frame=_env_value(
                source, "YOLO_VLM_GLOBAL_ORIGIN_FRAME", GLOBAL_ORIGIN_FRAME
            ),
            # Wrist RealSense intrinsics (measured from the running simulation)
            camera_fx=_env_float(source, "YOLO_VLM_CAMERA_FX", CAMERA_FX),
            camera_fy=_env_float(source, "YOLO_VLM_CAMERA_FY", CAMERA_FY),
            camera_cx=_env_float(source, "YOLO_VLM_CAMERA_CX", CAMERA_CX),
            camera_cy=_env_float(source, "YOLO_VLM_CAMERA_CY", CAMERA_CY),
            depth_patch_radius=int(_env_float(source, "YOLO_VLM_DEPTH_PATCH_RADIUS", DEPTH_PATCH_RADIUS)),
            targeted_scan_viewpoints=_env_int(
                source, "YOLO_VLM_TARGETED_SCAN_VIEWPOINTS", TARGETED_SCAN_VIEWPOINTS
            ),
            targeted_capture_fps=_env_float(
                source, "YOLO_VLM_TARGETED_CAPTURE_FPS", TARGETED_CAPTURE_FPS
            ),
            targeted_candidate_confidence=_env_float(
                source, "YOLO_VLM_TARGETED_CANDIDATE_CONFIDENCE", TARGETED_CANDIDATE_CONFIDENCE
            ),
            targeted_desk_width=_env_float(
                source, "YOLO_VLM_TARGETED_DESK_WIDTH", TARGETED_DESK_WIDTH
            ),
            targeted_table_scan_y=_env_float(
                source, "YOLO_VLM_TARGETED_TABLE_SCAN_Y", TARGETED_TABLE_SCAN_Y
            ),
            targeted_arc_radius=_env_float(
                source, "YOLO_VLM_TARGETED_ARC_RADIUS", TARGETED_ARC_RADIUS
            ),
            targeted_scan_height=_env_float(
                source, "YOLO_VLM_TARGETED_SCAN_HEIGHT", TARGETED_SCAN_HEIGHT
            ),
            targeted_desk_surface_z=_env_float(
                source,
                "YOLO_VLM_TARGETED_DESK_SURFACE_Z",
                TARGETED_DESK_SURFACE_Z,
            ),
            targeted_scan_roll=_env_float(
                source, "YOLO_VLM_TARGETED_SCAN_ROLL", TARGETED_SCAN_ROLL
            ),
            targeted_scan_pitch=_env_float(
                source, "YOLO_VLM_TARGETED_SCAN_PITCH", TARGETED_SCAN_PITCH
            ),
            targeted_close_standoff=_env_float(
                source, "YOLO_VLM_TARGETED_CLOSE_STANDOFF", TARGETED_CLOSE_STANDOFF
            ),
            targeted_cancel_timeout_sec=_env_float(
                source, "YOLO_VLM_TARGETED_CANCEL_TIMEOUT_SEC", TARGETED_CANCEL_TIMEOUT_SEC
            ),
        ),
        cartesian=CartesianConfig(
            command_timeout_sec=_env_float(
                source, "YOLO_VLM_CARTESIAN_COMMAND_TIMEOUT_SEC", CARTESIAN_COMMAND_TIMEOUT_SEC
            ),
            readiness_timeout_sec=_env_float(
                source, "YOLO_VLM_CARTESIAN_READY_TIMEOUT_SEC", CARTESIAN_READY_TIMEOUT_SEC
            ),
            tf_timeout_sec=_env_float(
                source, "YOLO_VLM_CARTESIAN_TF_TIMEOUT_SEC", CARTESIAN_TF_TIMEOUT_SEC
            ),
            base_frame=_env_value(
                source, "YOLO_VLM_CARTESIAN_BASE_FRAME", CARTESIAN_BASE_FRAME
            ),
            eef_frame=_env_value(
                source, "YOLO_VLM_CARTESIAN_EEF_FRAME", CARTESIAN_EEF_FRAME
            ),
        ),
        paths=PathConfig(
            static_semantic_coordinates=_env_path(
                source,
                "YOLO_VLM_STATIC_COORDINATES_PATH",
                STATIC_COORDINATES_PATH,
            ),
            dynamic_semantic_coordinates=_env_path(
                source,
                "YOLO_VLM_DYNAMIC_COORDINATES_PATH",
                DYNAMIC_COORDINATES_PATH,
            ),
            ros_setup_script=_env_path(
                source, "YOLO_VLM_ROS_SETUP_SCRIPT", ROS_SETUP_SCRIPT
            ),
            controller_workspace=controller_root,
            controller_setup_script=_env_path(
                source,
                "YOLO_VLM_CONTROLLER_SETUP_SCRIPT",
                controller_root / "install/setup.bash",
            ),
            pick_script=_env_path(
                source,
                "YOLO_VLM_PICK_SCRIPT",
                controller_root / PICK_SCRIPT_RELATIVE_PATH,
            ),
            place_script=_env_path(
                source,
                "YOLO_VLM_PLACE_SCRIPT",
                controller_root / PLACE_SCRIPT_RELATIVE_PATH,
            ),
        ),
        allow_mock_hardware_scripts=_env_bool(
            source, "YOLO_VLM_ALLOW_MOCK_HARDWARE_SCRIPTS", ALLOW_MOCK_HARDWARE_SCRIPTS
        ),
    )


def set_runtime_config(config: RuntimeConfig) -> None:
    """Install the validated configuration used by tools and the agent module."""
    global _runtime_config
    _runtime_config = config


def get_runtime_config() -> RuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = load_runtime_config()
    return _runtime_config


def _normalise_yolo_names(names: object) -> set[str]:
    if isinstance(names, Mapping):
        values = names.values()
    elif isinstance(names, Collection) and not isinstance(names, (str, bytes)):
        values = names
    else:
        raise RuntimeConfigurationError(
            "YOLO checkpoint did not expose a usable class-name vocabulary"
        )
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _load_yolo_class_names(checkpoint_path: Path) -> set[str]:
    """Read names from a local checkpoint; the path check prevents auto-downloads."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeConfigurationError(
            "The 'ultralytics' package is required to validate the local YOLO checkpoint"
        ) from exc

    try:
        model = YOLO(str(checkpoint_path), task="detect")
        return _normalise_yolo_names(model.names)
    except Exception as exc:
        raise RuntimeConfigurationError(
            f"Could not load YOLO checkpoint {checkpoint_path}: {exc}"
        ) from exc


def validate_yolo_target_class(
    config: RuntimeConfig,
    target: str,
    *,
    yolo_class_names: Collection[str] | Mapping[object, str] | None = None,
) -> None:
    """Fail clearly when a normalized target is absent from the local checkpoint."""
    normalized_target = target.strip().lower()
    if not normalized_target:
        raise RuntimeConfigurationError("YOLO target class must not be empty")
    if not config.yolo.checkpoint_path.is_file():
        raise RuntimeConfigurationError(
            f"YOLO checkpoint does not exist: {config.yolo.checkpoint_path}"
        )
    checkpoint_names = (
        _load_yolo_class_names(config.yolo.checkpoint_path)
        if yolo_class_names is None
        else _normalise_yolo_names(yolo_class_names)
    )
    if normalized_target not in checkpoint_names:
        raise RuntimeConfigurationError(
            f"YOLO checkpoint {config.yolo.checkpoint_path} does not provide "
            f"target class {normalized_target!r}"
        )


def validate_runtime_config(
    config: RuntimeConfig,
    *,
    yolo_class_names: Collection[str] | Mapping[object, str] | None = None,
) -> None:
    """Validate files, numeric safety limits, and checkpoint vocabulary."""
    errors: list[str] = []

    required_files = {
        "llama.cpp executable": config.vlm.executable,
        "VLM GGUF": config.vlm.model_path,
        "VLM multimodal projector": config.vlm.mmproj_path,
        "YOLO checkpoint": config.yolo.checkpoint_path,
        "static semantic coordinates": config.paths.static_semantic_coordinates,
        "ROS 2 setup script": config.paths.ros_setup_script,
        "controller setup script": config.paths.controller_setup_script,
    }
    for label, path in required_files.items():
        if not path.is_file():
            errors.append(f"{label} does not exist or is not a file: {path}")

    hardware_scripts = {
        "pick": config.paths.pick_script,
        "place": config.paths.place_script,
    }
    for action, path in hardware_scripts.items():
        if path.is_file():
            continue
        if config.allow_mock_hardware_scripts:
            print(
                f"[CONFIG WARNING] {action.capitalize()} hardware script is missing: "
                f"{path}. Mock hardware execution is enabled."
            )
        else:
            errors.append(
                f"{action} script does not exist or is not a file and mock "
                f"hardware execution is disabled: {path}"
            )

    if config.vlm.executable.is_file() and not os.access(config.vlm.executable, os.X_OK):
        errors.append(f"llama.cpp executable is not executable: {config.vlm.executable}")
    if not config.paths.controller_workspace.is_dir():
        errors.append(
            f"controller workspace does not exist or is not a directory: "
            f"{config.paths.controller_workspace}"
        )
    if config.paths.static_semantic_coordinates.is_file():
        try:
            with config.paths.static_semantic_coordinates.open(
                "r", encoding="utf-8"
            ) as semantic_file:
                semantic_coordinates = json.load(semantic_file)
            if not isinstance(semantic_coordinates, dict):
                errors.append("static semantic-coordinate JSON must contain an object")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"static semantic-coordinate JSON is invalid: {exc}")

    dynamic_path = config.paths.dynamic_semantic_coordinates
    if dynamic_path == config.paths.static_semantic_coordinates:
        errors.append(
            "dynamic and static semantic-coordinate paths must be different files"
        )
    if dynamic_path.suffix.lower() != ".json":
        errors.append(f"dynamic semantic-coordinate path must end in .json: {dynamic_path}")
    if not dynamic_path.parent.is_dir():
        errors.append(
            f"dynamic semantic-coordinate parent directory does not exist: {dynamic_path.parent}"
        )
    elif not os.access(dynamic_path.parent, os.W_OK):
        errors.append(
            f"dynamic semantic-coordinate parent directory is not writable: {dynamic_path.parent}"
        )
    if dynamic_path.exists() and not dynamic_path.is_file():
        errors.append(f"dynamic semantic-coordinate path is not a file: {dynamic_path}")
    elif dynamic_path.is_file() and not os.access(dynamic_path, os.W_OK):
        errors.append(f"dynamic semantic-coordinate file is not writable: {dynamic_path}")
    if dynamic_path.is_file():
        try:
            with dynamic_path.open("r", encoding="utf-8") as dynamic_file:
                dynamic_coordinates = json.load(dynamic_file)
            if not isinstance(dynamic_coordinates, dict):
                errors.append("dynamic semantic-coordinate JSON must contain an object")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"dynamic semantic-coordinate JSON is invalid: {exc}")

    if not config.vlm.host or any(char.isspace() for char in config.vlm.host):
        errors.append(f"server host is invalid: {config.vlm.host!r}")
    if not 1 <= config.vlm.port <= 65535:
        errors.append(f"server port must be between 1 and 65535, got {config.vlm.port}")
    if config.vlm.context_size != 4096:
        errors.append(
            "VLM context size must be exactly 4096 tokens for the 8GB "
            f"allocation, got {config.vlm.context_size}"
        )
    if config.vlm.parallel_slots != 1:
        errors.append(
            f"VLM parallel slots must be 1 for the 8GB allocation, got {config.vlm.parallel_slots}"
        )
    if config.vlm.gpu_layers != 99:
        errors.append(f"VLM GPU layers must be 99, got {config.vlm.gpu_layers}")
    if not config.vlm.flash_attention:
        errors.append("VLM Flash Attention must be enabled")
    if not 128 <= config.vlm.max_completion_tokens <= 256:
        errors.append(
            "VLM max completion tokens must be between 128 and 256, got "
            f"{config.vlm.max_completion_tokens}"
        )
    for label, value in (
        ("request timeout", config.vlm.request_timeout_sec),
        ("startup timeout", config.vlm.startup_timeout_sec),
        ("health timeout", config.vlm.health_timeout_sec),
    ):
        if value <= 0:
            errors.append(f"VLM {label} must be positive, got {value}")
    if config.vlm.vram_budget_gb <= 0:
        errors.append(f"VLM VRAM budget must be positive, got {config.vlm.vram_budget_gb}")
    if config.vlm.total_gpu_vram_gb <= 0:
        errors.append(
            f"total GPU VRAM must be positive, got {config.vlm.total_gpu_vram_gb}"
        )
    if config.vlm.vram_budget_gb > 8.0:
        errors.append(
            f"VLM VRAM budget must not exceed 8GB, got {config.vlm.vram_budget_gb}GB"
        )
    if config.vlm.vram_budget_gb > config.vlm.total_gpu_vram_gb:
        errors.append(
            "VLM VRAM budget cannot exceed total GPU VRAM "
            f"({config.vlm.vram_budget_gb}GB > {config.vlm.total_gpu_vram_gb}GB)"
        )
    managed_server_options = {
        "-m",
        "--model",
        "--mmproj",
        "-c",
        "--ctx-size",
        "-np",
        "--parallel",
        "-ngl",
        "--gpu-layers",
        "-fa",
        "--flash-attn",
        "-n",
        "--predict",
        "--n-predict",
        "--host",
        "--port",
        "--alias",
        "--jinja",
        "--no-jinja",
    }
    conflicting_options = sorted(
        option
        for option in config.vlm.extra_server_args
        if option.split("=", 1)[0] in managed_server_options
    )
    if conflicting_options:
        errors.append(
            "YOLO_VLM_SERVER_EXTRA_ARGS cannot override managed server options: "
            + ", ".join(conflicting_options)
        )

    if not (0.0 <= config.yolo.confidence_threshold <= 1.0):
        errors.append(
            "YOLO confidence threshold must be between 0.0 and 1.0, got "
            f"{config.yolo.confidence_threshold}"
        )
    if config.yolo.image_size <= 0:
        errors.append(f"YOLO image size must be positive, got {config.yolo.image_size}")
    if config.yolo.max_detections <= 0:
        errors.append(
            f"YOLO maximum detections must be positive, got {config.yolo.max_detections}"
        )

    rail_range = config.search.rail_max_position - config.search.rail_min_position
    if rail_range <= 0:
        errors.append(
            "rail minimum must be strictly less than rail maximum "
            f"({config.search.rail_min_position} >= {config.search.rail_max_position})"
        )
    if config.search.rail_waypoint_spacing <= 0 or (
        rail_range > 0 and config.search.rail_waypoint_spacing > rail_range
    ):
        errors.append(
            "rail waypoint spacing must be positive and no greater than the rail range, got "
            f"{config.search.rail_waypoint_spacing}"
        )
    if not math.isfinite(config.search.rail_speed) or config.search.rail_speed <= 0:
        errors.append(
            f"rail speed must be finite and positive, got {config.search.rail_speed}"
        )
    for angle in config.search.wrist_search_angles:
        if not -math.pi <= angle <= math.pi:
            errors.append(f"wrist angle must be within [-pi, pi] radians, got {angle}")
    if config.search.motion_settling_sec < 0:
        errors.append(
            f"motion settling duration cannot be negative, got {config.search.motion_settling_sec}"
        )
    if config.search.rail_joint_tolerance <= 0:
        errors.append(
            f"rail joint tolerance must be positive, got {config.search.rail_joint_tolerance}"
        )
    if config.search.wrist_joint_tolerance <= 0:
        errors.append(
            f"wrist joint tolerance must be positive, got {config.search.wrist_joint_tolerance}"
        )
    if config.search.targeted_scan_viewpoints < 2:
        errors.append("targeted scan viewpoints must be at least 2")
    for label, value in (
        ("targeted capture FPS", config.search.targeted_capture_fps),
        ("targeted desk width", config.search.targeted_desk_width),
        ("targeted table scan Y", config.search.targeted_table_scan_y),
        ("targeted arc radius", config.search.targeted_arc_radius),
        ("targeted scan height", config.search.targeted_scan_height),
        ("targeted close stand-off", config.search.targeted_close_standoff),
        ("targeted cancel timeout", config.search.targeted_cancel_timeout_sec),
    ):
        if not math.isfinite(value) or value <= 0:
            errors.append(f"{label} must be finite and positive, got {value}")
    if config.search.targeted_table_scan_y >= config.search.targeted_desk_width:
        errors.append(
            "targeted table scan Y must be inside the desk edge, got "
            f"{config.search.targeted_table_scan_y} >= "
            f"{config.search.targeted_desk_width}"
        )
    if not math.isfinite(config.search.targeted_desk_surface_z):
        errors.append(
            "targeted desk surface Z must be finite, got "
            f"{config.search.targeted_desk_surface_z}"
        )
    if not 0 < config.search.targeted_candidate_confidence < config.yolo.confidence_threshold:
        errors.append(
            "targeted candidate confidence must be between 0 and the final YOLO "
            f"threshold ({config.yolo.confidence_threshold}), got "
            f"{config.search.targeted_candidate_confidence}"
        )
    for label, angle in (
        ("targeted scan roll", config.search.targeted_scan_roll),
        ("targeted scan pitch", config.search.targeted_scan_pitch),
    ):
        if not math.isfinite(angle) or not -math.pi <= angle <= math.pi:
            errors.append(f"{label} must be within [-pi, pi], got {angle}")
    for frame_name, frame_value in (
        ("camera base frame", config.search.camera_base_frame),
        ("camera optical frame", config.search.camera_optical_frame),
        ("global origin frame", config.search.global_origin_frame),
    ):
        if not frame_value.strip():
            errors.append(f"{frame_name} must not be empty")
    if config.search.global_origin_frame == config.search.camera_optical_frame:
        errors.append("global origin frame must differ from the camera optical frame")
    for label, value in (
        ("Cartesian command timeout", config.cartesian.command_timeout_sec),
        ("Cartesian readiness timeout", config.cartesian.readiness_timeout_sec),
        ("Cartesian TF timeout", config.cartesian.tf_timeout_sec),
    ):
        if value <= 0:
            errors.append(f"{label} must be positive, got {value}")

    if not errors and config.yolo.checkpoint_path.is_file():
        try:
            checkpoint_names = (
                _load_yolo_class_names(config.yolo.checkpoint_path)
                if yolo_class_names is None
                else _normalise_yolo_names(yolo_class_names)
            )
            missing_classes = sorted(
                set(config.yolo.required_classes) - checkpoint_names
            )
            if missing_classes:
                errors.append(
                    "YOLO checkpoint is missing required target class(es): "
                    + ", ".join(missing_classes)
                )
        except RuntimeConfigurationError as exc:
            errors.append(str(exc))

    if errors:
        formatted = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeConfigurationError(f"Runtime configuration validation failed:\n{formatted}")
