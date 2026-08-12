"""Pluggable visual-appearance embeddings for semantic association."""

from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .detector import RawDetection


class AppearanceEmbeddingProvider(Protocol):
    """Supplies one L2-normalized embedding for each detection crop."""

    @property
    def provider_id(self) -> str:
        ...

    def attach(self, image: np.ndarray, detections: list[RawDetection]) -> list[RawDetection]:
        ...


class DisabledAppearanceEmbeddingProvider:
    @property
    def provider_id(self) -> str:
        return "disabled"

    def attach(self, image: np.ndarray, detections: list[RawDetection]) -> list[RawDetection]:
        return detections


class UltralyticsCropAppearanceEmbeddingProvider:
    """Extract penultimate-layer YOLO embeddings from detection crops."""

    def __init__(self, model: Any, model_id: str, device: str, image_size: int) -> None:
        self._model = model
        self._device = device
        self._image_size = image_size
        self._provider_id = f"ultralytics_crop:{model_id}"

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def attach(self, image: np.ndarray, detections: list[RawDetection]) -> list[RawDetection]:
        if not detections:
            return detections
        crops: list[np.ndarray] = []
        crop_indices: list[int] = []
        height, width = image.shape[:2]
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox_xyxy
            left = max(0, min(width, int(np.floor(x1))))
            right = max(0, min(width, int(np.ceil(x2))))
            top = max(0, min(height, int(np.floor(y1))))
            bottom = max(0, min(height, int(np.ceil(y2))))
            if right <= left or bottom <= top:
                continue
            crops.append(np.ascontiguousarray(image[top:bottom, left:right]))
            crop_indices.append(index)
        if not crops:
            return detections
        try:
            embeddings = self._model.embed(
                crops,
                imgsz=self._image_size,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"YOLO crop embedding failed: {exc}") from exc
        updated = list(detections)
        for detection_index, embedding in zip(crop_indices, embeddings):
            vector = embedding.detach().cpu().numpy().astype(np.float64, copy=False)
            vector = vector.reshape(-1)
            norm = float(np.linalg.norm(vector))
            if vector.size == 0 or not np.all(np.isfinite(vector)) or norm <= 0.0:
                continue
            updated[detection_index] = replace(
                updated[detection_index],
                appearance_embedding=(vector / norm).astype(np.float32),
                appearance_provider_id=self.provider_id,
            )
        return updated


class MobileNetV3SmallAppearanceEmbeddingProvider:
    """ImageNet MobileNetV3-Small crop embeddings projected to 128 dimensions."""

    _PROJECTION_SEED = 20260811
    _EMBEDDING_DIMENSION = 128

    def __init__(self, device: str, image_size: int) -> None:
        try:
            import torch
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as exc:
            raise RuntimeError(
                "mobilenet_v3_small appearance embeddings require torch and torchvision"
            ) from exc
        self._torch = torch
        self._device = torch.device(device)
        self._image_size = image_size
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self._model = mobilenet_v3_small(weights=weights).features.to(self._device).eval()
        generator = np.random.default_rng(self._PROJECTION_SEED)
        projection = generator.standard_normal((576, self._EMBEDDING_DIMENSION))
        projection /= np.linalg.norm(projection, axis=0, keepdims=True)
        self._projection = projection.astype(np.float32)
        self._provider_id = "mobilenet_v3_small:imagenet1k:rp128-v1"

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def attach(self, image: np.ndarray, detections: list[RawDetection]) -> list[RawDetection]:
        if not detections:
            return detections
        crops: list[np.ndarray] = []
        crop_indices: list[int] = []
        height, width = image.shape[:2]
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox_xyxy
            left = max(0, min(width, int(np.floor(x1))))
            right = max(0, min(width, int(np.ceil(x2))))
            top = max(0, min(height, int(np.floor(y1))))
            bottom = max(0, min(height, int(np.ceil(y2))))
            if right <= left or bottom <= top:
                continue
            crop = image[top:bottom, left:right]
            crop = cv2.resize(crop, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA)
            crops.append(crop)
            crop_indices.append(index)
        if not crops:
            return detections
        batch = np.stack(crops).astype(np.float32) / 255.0
        # Detector images are BGR; ImageNet model expects RGB and normalized input.
        batch = batch[..., ::-1].transpose(0, 3, 1, 2).copy()
        batch -= np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
        batch /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
        try:
            tensor = self._torch.from_numpy(batch).to(self._device)
            with self._torch.inference_mode():
                features = self._model(tensor)
                features = self._torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
            embeddings = features.cpu().numpy() @ self._projection
        except Exception as exc:
            raise RuntimeError(f"MobileNet crop embedding failed: {exc}") from exc
        updated = list(detections)
        for detection_index, vector in zip(crop_indices, embeddings):
            norm = float(np.linalg.norm(vector))
            if not np.all(np.isfinite(vector)) or norm <= 0.0:
                continue
            updated[detection_index] = replace(
                updated[detection_index],
                appearance_embedding=(vector / norm).astype(np.float32),
                appearance_provider_id=self.provider_id,
            )
        return updated


class MobileCLIPS0AppearanceEmbeddingProvider:
    """Batched image-only MobileCLIP-S0 embeddings for YOLO crops."""

    def __init__(self, device: str, checkpoint_path: str) -> None:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint_path.strip():
            raise RuntimeError(
                "appearance.mobileclip_checkpoint must be set to a local "
                "MobileCLIP-S0 .pt checkpoint"
            )
        if not checkpoint.is_file():
            raise RuntimeError(
                f"MobileCLIP-S0 checkpoint does not exist: {checkpoint}"
            )
        try:
            import torch
            import mobileclip
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "mobileclip_s0 appearance embeddings require Apple's mobileclip package; "
                "install it from https://github.com/apple/ml-mobileclip"
            ) from exc

        requested_device = torch.device(device)
        self._device = (
            requested_device
            if requested_device.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        try:
            model, _, preprocess = mobileclip.create_model_and_transforms(
                "mobileclip_s0", pretrained=str(checkpoint)
            )
            # The official loader reparameterizes by default. Calling its helper
            # again corrupts already-folded RepMixer modules.
            self._model = model.to(self._device).eval()
        except Exception as exc:
            raise RuntimeError(f"Could not load MobileCLIP-S0 checkpoint {checkpoint}: {exc}") from exc
        self._torch = torch
        self._image_class = Image
        self._preprocess = preprocess
        stat = checkpoint.stat()
        self._provider_id = (
            f"mobileclip_s0:{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def attach(self, image: np.ndarray, detections: list[RawDetection]) -> list[RawDetection]:
        if not detections:
            return detections
        crops: list[Any] = []
        crop_indices: list[int] = []
        height, width = image.shape[:2]
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox_xyxy
            left = max(0, min(width, int(np.floor(x1))))
            right = max(0, min(width, int(np.ceil(x2))))
            top = max(0, min(height, int(np.floor(y1))))
            bottom = max(0, min(height, int(np.ceil(y2))))
            if right <= left or bottom <= top:
                continue
            # The detector frame is BGR.  Convert only the cropped view for the
            # official MobileCLIP transform; all valid crops are encoded together.
            crop_rgb = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            crops.append(self._preprocess(self._image_class.fromarray(crop_rgb)))
            crop_indices.append(index)
        if not crops:
            return detections

        try:
            batch = self._torch.stack(crops, dim=0).to(self._device, non_blocking=True)
            autocast = (
                self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
                if self._device.type == "cuda"
                else _null_context()
            )
            with self._torch.inference_mode(), autocast:
                embeddings = self._model.encode_image(batch)
                embeddings = self._torch.nn.functional.normalize(
                    embeddings.float(), p=2, dim=-1
                )
            vectors = embeddings.cpu().numpy().astype(np.float32, copy=False)
        except Exception as exc:
            raise RuntimeError(f"MobileCLIP-S0 crop embedding failed: {exc}") from exc

        updated = list(detections)
        for detection_index, vector in zip(crop_indices, vectors):
            if vector.size == 0 or not np.all(np.isfinite(vector)):
                continue
            updated[detection_index] = replace(
                updated[detection_index],
                appearance_embedding=vector.reshape(-1),
                appearance_provider_id=self.provider_id,
            )
        return updated


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> bool:
        return False


def create_appearance_provider(
    enabled: bool,
    backend: str,
    detector: Any,
    device: str,
    image_size: int,
    mobileclip_checkpoint: str = "",
) -> AppearanceEmbeddingProvider:
    if not enabled or backend == "disabled":
        return DisabledAppearanceEmbeddingProvider()
    if backend == "mobilenet_v3_small":
        return MobileNetV3SmallAppearanceEmbeddingProvider(device, image_size)
    if backend == "mobileclip_s0":
        return MobileCLIPS0AppearanceEmbeddingProvider(device, mobileclip_checkpoint)
    if backend == "ultralytics_crop" and hasattr(detector, "_model"):
        return UltralyticsCropAppearanceEmbeddingProvider(
            detector._model, detector.model_id, device, image_size
        )
    raise RuntimeError(
        f"Unsupported appearance backend {backend!r}; use mobileclip_s0, "
        "mobilenet_v3_small, ultralytics_crop, or disabled"
    )
