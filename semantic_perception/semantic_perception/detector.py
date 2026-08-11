"""Detector-neutral inference contract and Ultralytics adapter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RawDetection:
    class_name: str
    confidence: float
    class_likelihoods: dict[str, float]
    bbox_xyxy: tuple[float, float, float, float]


class ObjectDetector(Protocol):
    @property
    def model_id(self) -> str:
        ...

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        ...


class UltralyticsDetector:
    def __init__(
        self,
        model_path: str,
        device: str,
        image_size: int,
        confidence_threshold: float,
        max_detections: int,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Detector model does not exist: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("The ultralytics package is required") from exc
        self._model = YOLO(str(path), task="detect")
        self._model_id = path.name
        self._device = device
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        self._max_detections = max_detections

    @property
    def model_id(self) -> str:
        return self._model_id

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        results = self._model.predict(
            source=image,
            imgsz=self._image_size,
            conf=self._confidence_threshold,
            device=self._device,
            max_det=self._max_detections,
            verbose=False,
        )
        detections: list[RawDetection] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        names = getattr(result, "names", getattr(self._model, "names", {}))
        for coordinates, confidence, class_id in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
        ):
            numeric_id = int(class_id)
            if isinstance(names, dict):
                label = str(names.get(numeric_id, numeric_id))
            else:
                label = str(names[numeric_id])
            detections.append(
                RawDetection(
                    class_name=label.strip().lower().replace(" ", "_"),
                    confidence=float(confidence),
                    class_likelihoods={
                        label.strip().lower().replace(" ", "_"): float(confidence),
                        "other": 1.0 - float(confidence),
                    },
                    bbox_xyxy=tuple(float(value) for value in coordinates),
                )
            )
        return detections


def create_detector(
    backend: str,
    model_path: str,
    device: str,
    image_size: int,
    confidence_threshold: float,
    max_detections: int,
) -> ObjectDetector:
    if backend == "ultralytics":
        return UltralyticsDetector(
            model_path=model_path,
            device=device,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
        )
    raise RuntimeError(f"Unsupported detector backend: {backend!r}")
