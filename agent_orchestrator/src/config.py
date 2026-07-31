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
YOLO_CHECKPOINT_PATH: Path | None = PROJECT_ROOT / "agent_orchestrator/models/yolo11s.pt"


# --- VLM server settings ---
VLM_SERVER_HOST            = "127.0.0.1"
VLM_SERVER_PORT            = 8080
VLM_CONTEXT_SIZE           = 4096
VLM_PARALLEL_SLOTS         = 1
VLM_GPU_LAYERS             = 99
VLM_FLASH_ATTENTION        = True
VLM_MAX_COMPLETION_TOKENS  = 256
VLM_VRAM_BUDGET_GB         = 8.0
VLM_TOTAL_GPU_VRAM_GB      = 16.0

# --- Search / rail parameters ---
SEARCH_RAIL_MIN             = -1.30  # metres
SEARCH_RAIL_MAX             =  1.60   # metres
SEARCH_RAIL_WAYPOINT_SPACING = 0.40  # metres between scan stops

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
    centering_min_step: float
    centering_max_step: float
    centering_gain: float
    wrist_search_angles: tuple[float, ...]
    final_centering_angle: float
    motion_settling_sec: float
    rail_joint_tolerance: float
    wrist_joint_tolerance: float
    horizontal_center_tolerance: float


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
                source, "YOLO_VLM_REQUEST_TIMEOUT_SEC", 120.0
            ),
            startup_timeout_sec=_env_float(
                source, "YOLO_VLM_STARTUP_TIMEOUT_SEC", 45.0
            ),
            health_timeout_sec=_env_float(
                source, "YOLO_VLM_HEALTH_TIMEOUT_SEC", 2.0
            ),
            vram_budget_gb=_env_float(source, "YOLO_VLM_VRAM_BUDGET_GB", VLM_VRAM_BUDGET_GB),
            total_gpu_vram_gb=_env_float(
                source, "YOLO_VLM_TOTAL_GPU_VRAM_GB", VLM_TOTAL_GPU_VRAM_GB
            ),
            extra_server_args=tuple(
                shlex.split(source.get("YOLO_VLM_SERVER_EXTRA_ARGS", ""))
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
                source, "YOLO_VLM_YOLO_CONFIDENCE", 0.85
            ),
            image_size=_env_int(source, "YOLO_VLM_YOLO_IMAGE_SIZE", 640),
            device=_env_value(source, "YOLO_VLM_YOLO_DEVICE", "cuda:0"),
            max_detections=_env_int(source, "YOLO_VLM_YOLO_MAX_DETECTIONS", 100),
            required_classes=_env_string_list(
                source, "YOLO_VLM_YOLO_REQUIRED_CLASSES", ("apple",)
            ),
        ),
        search=SearchConfig(
            rail_min_position=_env_float(source, "YOLO_VLM_RAIL_MIN", SEARCH_RAIL_MIN),
            rail_max_position=_env_float(source, "YOLO_VLM_RAIL_MAX", SEARCH_RAIL_MAX),
            rail_waypoint_spacing=_env_float(
                source, "YOLO_VLM_RAIL_WAYPOINT_SPACING", SEARCH_RAIL_WAYPOINT_SPACING
            ),
            centering_min_step=_env_float(
                source, "YOLO_VLM_CENTERING_MIN_STEP", 0.005
            ),
            centering_max_step=_env_float(
                source, "YOLO_VLM_CENTERING_MAX_STEP", 0.10
            ),
            centering_gain=_env_float(source, "YOLO_VLM_CENTERING_GAIN", 0.10),
            wrist_search_angles=_env_float_list(
                source, "YOLO_VLM_WRIST_SEARCH_ANGLES", (-1.57, 0.0, 1.57)
            ),
            final_centering_angle=_env_float(
                source, "YOLO_VLM_FINAL_CENTERING_ANGLE", 1.57
            ),
            motion_settling_sec=_env_float(
                source, "YOLO_VLM_MOTION_SETTLING_SEC", 0.50
            ),
            rail_joint_tolerance=_env_float(
                source, "YOLO_VLM_RAIL_JOINT_TOLERANCE", 0.05
            ),
            wrist_joint_tolerance=_env_float(
                source, "YOLO_VLM_WRIST_JOINT_TOLERANCE", 0.05
            ),
            horizontal_center_tolerance=_env_float(
                source, "YOLO_VLM_HORIZONTAL_CENTER_TOLERANCE", 0.05
            ),
        ),
        paths=PathConfig(
            static_semantic_coordinates=_env_path(
                source,
                "YOLO_VLM_STATIC_COORDINATES_PATH",
                PROJECT_ROOT / "semantic_distances.json",
            ),
            dynamic_semantic_coordinates=_env_path(
                source,
                "YOLO_VLM_DYNAMIC_COORDINATES_PATH",
                PROJECT_ROOT / "semantic_distances_dynamic.json",
            ),
            ros_setup_script=_env_path(
                source, "YOLO_VLM_ROS_SETUP_SCRIPT", Path("/opt/ros/humble/setup.bash")
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
                controller_root / "src/bringup/rail_demo_pick.sh",
            ),
            place_script=_env_path(
                source,
                "YOLO_VLM_PLACE_SCRIPT",
                controller_root / "src/bringup/rail_demo_place.sh",
            ),
        ),
        allow_mock_hardware_scripts=_env_bool(
            source, "YOLO_VLM_ALLOW_MOCK_HARDWARE_SCRIPTS", True
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

    if config.yolo.confidence_threshold != 0.85:
        errors.append(
            "YOLO confidence threshold must be exactly 0.85, got "
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
    if config.search.centering_min_step <= 0:
        errors.append(
            f"centering minimum step must be positive, got {config.search.centering_min_step}"
        )
    if config.search.centering_max_step < config.search.centering_min_step or (
        rail_range > 0 and config.search.centering_max_step > rail_range
    ):
        errors.append(
            "centering maximum step must be at least the minimum step and within the rail range"
        )
    if config.search.centering_gain <= 0:
        errors.append(f"centering gain must be positive, got {config.search.centering_gain}")
    for angle in (*config.search.wrist_search_angles, config.search.final_centering_angle):
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
    if not 0 < config.search.horizontal_center_tolerance < 1:
        errors.append(
            "horizontal centering tolerance must be between 0 and 1, got "
            f"{config.search.horizontal_center_tolerance}"
        )

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
