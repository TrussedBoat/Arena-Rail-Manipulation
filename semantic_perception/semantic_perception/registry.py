"""Probabilistic data association and durable semantic object registry."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import linear_sum_assignment

from .localization import LocalizedDetection


OTHER_CLASS = "other"
UNKNOWN_CLASS = "unknown"


@dataclass(frozen=True)
class RegistryConfig:
    mahalanobis_threshold: float = 11.345
    confirmation_hits: int = 3
    confirmation_window_sec: float = 3.0
    class_confirmation_probability: float = 0.70
    max_explicit_classes: int = 4
    stale_after_sec: float = 120.0
    stale_retention_sec: float = 120.0
    evidence_decay: float = 0.95
    process_noise_stddev_m: float = 0.002
    appearance_max_cosine_distance: float = 0.35
    appearance_cost_weight: float = 0.5
    appearance_embedding_decay: float = 0.90
    distance_evidence_reference_m: float = 1.0
    distance_evidence_min_weight: float = 0.25


@dataclass
class ObjectTrack:
    object_id: str
    position: np.ndarray
    position_covariance: np.ndarray
    observation_count: int
    first_seen_sec: float
    last_seen_sec: float
    model_id: str
    appearance_embedding: np.ndarray | None = None
    appearance_provider_id: str | None = None
    class_scores: dict[str, float] = field(default_factory=dict)
    recent_observations: deque[tuple[float, dict[str, float]]] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    state: str = "candidate"
    confirmation_locked: bool = False
    stale_since_sec: float | None = None

    @property
    def class_distribution(self) -> dict[str, float]:
        total = sum(max(0.0, value) for value in self.class_scores.values())
        if total <= 0.0:
            return {OTHER_CLASS: 1.0}
        distribution = {
            label: max(0.0, value) / total
            for label, value in self.class_scores.items()
            if value > 0.0
        }
        distribution.setdefault(OTHER_CLASS, 0.0)
        return distribution

    @property
    def class_name(self) -> str:
        distribution = self.class_distribution
        winner = max(distribution, key=lambda label: (distribution[label], label))
        return UNKNOWN_CLASS if winner == OTHER_CLASS else winner

    @property
    def class_probability(self) -> float:
        distribution = self.class_distribution
        if self.class_name == UNKNOWN_CLASS:
            return distribution.get(OTHER_CLASS, 0.0)
        return distribution.get(self.class_name, 0.0)

    @property
    def confidence(self) -> float:
        """Compatibility alias for the representative class probability."""
        return self.class_probability

    @property
    def position_stddev_m(self) -> float:
        return float(np.sqrt(np.max(np.diag(self.position_covariance))))


@dataclass(frozen=True)
class AssociationResult:
    detection_index: int
    track: ObjectTrack
    decision: str
    eligible_track_count: int
    best_mahalanobis_d2: float | None
    selected_mahalanobis_d2: float | None
    selected_cost: float | None


class ObjectRegistry:
    def __init__(self, config: RegistryConfig, model_id: str) -> None:
        self.config = config
        self.model_id = model_id
        self._tracks: dict[str, ObjectTrack] = {}
        self._next_object_id = 0

    def tracks(self, now_sec: float | None = None) -> list[ObjectTrack]:
        return sorted(self._tracks.values(), key=lambda track: track.object_id)

    def maintain(self, now_sec: float) -> tuple[list[str], list[str], list[str]]:
        """Advance lifecycle state; return candidates expired, tracks marked stale, tracks deleted."""
        expired_candidates = self.expire_candidates(now_sec)
        marked_stale: list[str] = []
        deleted_stale: list[str] = []
        for track in self._tracks.values():
            unseen_sec = now_sec - track.last_seen_sec
            if track.state != "stale" and unseen_sec > self.config.stale_after_sec:
                track.state = "stale"
                track.stale_since_sec = now_sec
                marked_stale.append(track.object_id)
            elif track.state == "stale" and track.stale_since_sec is None:
                # Older registries have no transition time. Start their
                # retention period now instead of deleting them immediately.
                track.stale_since_sec = now_sec
                marked_stale.append(track.object_id)
            if track.state == "stale" and track.stale_since_sec is not None and (
                now_sec - track.stale_since_sec > self.config.stale_retention_sec
            ):
                deleted_stale.append(track.object_id)
        for object_id in deleted_stale:
            del self._tracks[object_id]
        return expired_candidates, marked_stale, deleted_stale

    def expire_candidates(self, now_sec: float) -> list[str]:
        """Remove candidates that failed to confirm within their hit window."""
        expired = [
            track.object_id
            for track in self._tracks.values()
            if (
                track.state == "candidate"
                and now_sec - track.last_seen_sec > self.config.confirmation_window_sec
            )
        ]
        for object_id in expired:
            del self._tracks[object_id]
        return expired

    def update(self, detection: LocalizedDetection) -> ObjectTrack:
        """Compatibility helper for a frame containing one detection."""
        return self.update_frame([detection])[0]

    def update_frame(
        self, detections: list[LocalizedDetection]
    ) -> list[ObjectTrack]:
        """Associate one RGB-D frame one-to-one, then update/create tracks."""
        return [result.track for result in self.update_frame_with_diagnostics(detections)]

    def update_frame_with_diagnostics(
        self, detections: list[LocalizedDetection]
    ) -> list[AssociationResult]:
        """Associate a frame and retain costs needed for telemetry."""
        if not detections:
            return []
        for detection in detections:
            self._validate_detection(detection)

        tracks = list(self._tracks.values())
        costs = np.full((len(tracks), len(detections)), np.inf, dtype=np.float64)
        distances = np.full((len(tracks), len(detections)), np.nan, dtype=np.float64)
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                distance = self._mahalanobis_distance_squared(track, detection)
                if distance is not None:
                    distances[track_index, detection_index] = distance
                cost = self._association_cost(track, detection)
                if cost is not None:
                    costs[track_index, detection_index] = cost

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        result_by_detection: dict[int, AssociationResult] = {}
        for track_index, detection_index in self._hungarian_matches(costs):
            track = tracks[track_index]
            self._update_track(track, detections[detection_index])
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            result_by_detection[detection_index] = AssociationResult(
                detection_index=detection_index,
                track=track,
                decision="matched",
                eligible_track_count=int(np.isfinite(costs[:, detection_index]).sum()),
                best_mahalanobis_d2=_finite_min(distances[:, detection_index]),
                selected_mahalanobis_d2=float(distances[track_index, detection_index]),
                selected_cost=float(costs[track_index, detection_index]),
            )

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = self._new_track(detection)
            self._tracks[track.object_id] = track
            result_by_detection[detection_index] = AssociationResult(
                detection_index=detection_index,
                track=track,
                decision="new_track",
                eligible_track_count=int(np.isfinite(costs[:, detection_index]).sum()),
                best_mahalanobis_d2=_finite_min(distances[:, detection_index]),
                selected_mahalanobis_d2=None,
                selected_cost=None,
            )

        return [result_by_detection[index] for index in range(len(detections))]

    def merge_confirmed_duplicates(self) -> list[tuple[str, str]]:
        """Conservatively merge spatially and visually matching confirmed tracks."""
        merged: list[tuple[str, str]] = []
        while True:
            tracks = sorted(self._tracks.values(), key=lambda track: track.object_id)
            pair: tuple[ObjectTrack, ObjectTrack] | None = None
            for index, left in enumerate(tracks):
                if left.state != "confirmed" or left.class_name == UNKNOWN_CLASS:
                    continue
                for right in tracks[index + 1 :]:
                    if (
                        right.state == "confirmed"
                        and right.class_name == left.class_name
                        and self._confirmed_tracks_match(left, right)
                    ):
                        pair = left, right
                        break
                if pair is not None:
                    break
            if pair is None:
                return merged
            survivor, duplicate = pair
            self._merge_tracks(survivor, duplicate)
            del self._tracks[duplicate.object_id]
            merged.append((survivor.object_id, duplicate.object_id))

    def _confirmed_tracks_match(self, left: ObjectTrack, right: ObjectTrack) -> bool:
        innovation = right.position - left.position
        covariance = left.position_covariance + right.position_covariance
        try:
            distance_squared = float(innovation.T @ np.linalg.solve(covariance, innovation))
        except np.linalg.LinAlgError:
            return False
        if not np.isfinite(distance_squared) or distance_squared > self.config.mahalanobis_threshold:
            return False
        appearance_distance = self._track_appearance_distance(left, right)
        return (
            appearance_distance is not None
            and appearance_distance <= self.config.appearance_max_cosine_distance
        )

    @staticmethod
    def _track_appearance_distance(left: ObjectTrack, right: ObjectTrack) -> float | None:
        if (
            left.appearance_embedding is None
            or right.appearance_embedding is None
            or not left.appearance_provider_id
            or left.appearance_provider_id != right.appearance_provider_id
            or left.appearance_embedding.shape != right.appearance_embedding.shape
        ):
            return None
        distance = 1.0 - float(np.dot(left.appearance_embedding, right.appearance_embedding))
        return float(np.clip(distance, 0.0, 2.0)) if np.isfinite(distance) else None

    def _merge_tracks(self, survivor: ObjectTrack, duplicate: ObjectTrack) -> None:
        """Fuse correlated estimates using equal-weight covariance intersection."""
        inverse_survivor = np.linalg.inv(survivor.position_covariance)
        inverse_duplicate = np.linalg.inv(duplicate.position_covariance)
        covariance = np.linalg.inv(0.5 * inverse_survivor + 0.5 * inverse_duplicate)
        survivor.position = covariance @ (
            0.5 * inverse_survivor @ survivor.position
            + 0.5 * inverse_duplicate @ duplicate.position
        )
        survivor.position_covariance = (covariance + covariance.T) / 2.0
        survivor.observation_count += duplicate.observation_count
        survivor.first_seen_sec = min(survivor.first_seen_sec, duplicate.first_seen_sec)
        survivor.last_seen_sec = max(survivor.last_seen_sec, duplicate.last_seen_sec)
        survivor.stale_since_sec = None
        for label, score in duplicate.class_scores.items():
            survivor.class_scores[label] = survivor.class_scores.get(label, 0.0) + score
        self._prune_class_scores(survivor)
        survivor.recent_observations = deque(
            sorted(
                (*survivor.recent_observations, *duplicate.recent_observations),
                key=lambda observation: observation[0],
            )[-100:],
            maxlen=100,
        )
        if self._track_appearance_distance(survivor, duplicate) is not None:
            survivor_weight = max(1, survivor.observation_count - duplicate.observation_count)
            duplicate_weight = max(1, duplicate.observation_count)
            survivor.appearance_embedding = _normalized_embedding(
                survivor_weight * survivor.appearance_embedding
                + duplicate_weight * duplicate.appearance_embedding
            )
        survivor.confirmation_locked = True
        self._update_state(survivor, survivor.last_seen_sec)

    @staticmethod
    def _hungarian_matches(costs: np.ndarray) -> list[tuple[int, int]]:
        """Globally minimize eligible matching costs while allowing unmatched rows."""
        track_count, detection_count = costs.shape
        if track_count == 0 or detection_count == 0:
            return []
        # Every eligible association cost is normalized to <= 1. Give each
        # unmatched item cost 1, so any eligible edge beats leaving both ends
        # unmatched. Invalid edges are always more expensive than that choice.
        unmatched_cost = 1.0
        invalid_cost = 4.0
        matrix_size = track_count + detection_count
        matrix = np.full((matrix_size, matrix_size), invalid_cost, dtype=np.float64)
        matrix[:track_count, :detection_count] = np.where(
            np.isfinite(costs), costs, invalid_cost
        )
        for track_index in range(track_count):
            matrix[track_index, detection_count + track_index] = unmatched_cost
        for detection_index in range(detection_count):
            matrix[track_count + detection_index, detection_index] = unmatched_cost
        matrix[track_count:, detection_count:] = 0.0
        rows, columns = linear_sum_assignment(matrix)
        return [
            (int(track_index), int(detection_index))
            for track_index, detection_index in zip(rows, columns)
            if track_index < track_count
            and detection_index < detection_count
            and np.isfinite(costs[track_index, detection_index])
        ]

    def best_confirmed(
        self,
        class_name: str,
        min_confidence: float,
        seen_since_sec: float | None = None,
    ) -> ObjectTrack | None:
        target = _normalize_label(class_name)
        required_probability = max(
            min_confidence, self.config.class_confirmation_probability
        )
        matches: list[ObjectTrack] = []
        for track in self._tracks.values():
            distribution = track.class_distribution
            if (
                track.class_name != target
                or track.state != "confirmed"
                or distribution.get(target, 0.0) < required_probability
            ):
                continue
            if seen_since_sec is not None:
                fresh_count = sum(
                    stamp >= seen_since_sec
                    for stamp, _ in track.recent_observations
                )
                if fresh_count < self.config.confirmation_hits:
                    continue
            matches.append(track)
        if not matches:
            return None
        return max(
            matches,
            key=lambda track: (track.class_probability, track.last_seen_sec),
        )

    def best_candidate(self, class_name: str) -> ObjectTrack | None:
        """Return a still-unconfirmed candidate for FindObject feedback.

        ``class_ambiguous`` tracks may retain weak probability for a class, but
        they are not short-lived candidate observations and must not pause a
        targeted scan indefinitely.
        """
        target = _normalize_label(class_name)
        matches = [
            track
            for track in self._tracks.values()
            if (
                track.state == "candidate"
                and track.class_distribution.get(target, 0.0) > 0.0
            )
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda track: (
                track.class_distribution.get(target, 0.0),
                track.last_seen_sec,
            ),
        )

    def load(self, path: Path) -> None:
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and isinstance(payload.get("objects"), list):
            schema_version = int(payload.get("schema_version", 1))
            for item in payload["objects"]:
                self._load_versioned_item(item, schema_version)
            return
        if isinstance(payload, dict):
            self._load_legacy_coordinates(payload)
            return
        raise ValueError(f"Registry JSON must contain an object: {path}")

    def persist(self, path: Path, reference_frame: str) -> None:
        payload = {
            "schema_version": 4,
            "reference_frame": reference_frame,
            "updated_at": _iso_time(datetime.now(tz=timezone.utc).timestamp()),
            "objects": [self._serialize_track(track) for track in self.tracks()],
        }
        _atomic_json_write(path, payload)

    def persist_legacy_coordinates(self, path: Path) -> None:
        by_class: dict[str, ObjectTrack] = {}
        for track in self._tracks.values():
            if track.state != "confirmed" or track.class_name == UNKNOWN_CLASS:
                continue
            current = by_class.get(track.class_name)
            if current is None or (
                track.class_probability,
                track.last_seen_sec,
            ) > (current.class_probability, current.last_seen_sec):
                by_class[track.class_name] = track
        payload = {
            label: {
                "x": round(float(track.position[0]), 4),
                "y": round(float(track.position[1]), 4),
                "z": round(float(track.position[2]), 4),
            }
            for label, track in sorted(by_class.items())
        }
        _atomic_json_write(path, payload)

    def _mahalanobis_distance_squared(
        self, track: ObjectTrack, detection: LocalizedDetection
    ) -> float | None:
        process_covariance = np.eye(3) * self.config.process_noise_stddev_m**2
        predicted_covariance = track.position_covariance + process_covariance
        innovation_covariance = predicted_covariance + detection.position_covariance
        innovation = np.asarray(detection.position, dtype=np.float64) - track.position
        try:
            solution = np.linalg.solve(innovation_covariance, innovation)
        except np.linalg.LinAlgError:
            return None
        distance = float(innovation.T @ solution)
        return distance if np.isfinite(distance) and distance >= 0.0 else None

    def _association_cost(
        self, track: ObjectTrack, detection: LocalizedDetection
    ) -> float | None:
        distance_squared = self._mahalanobis_distance_squared(track, detection)
        if (
            distance_squared is None
            or distance_squared > self.config.mahalanobis_threshold
        ):
            return None
        spatial_cost = distance_squared / self.config.mahalanobis_threshold
        appearance_distance = self._appearance_distance(track, detection)
        if appearance_distance is None:
            return spatial_cost
        if appearance_distance > self.config.appearance_max_cosine_distance:
            return None
        weight = self.config.appearance_cost_weight
        return (1.0 - weight) * spatial_cost + weight * appearance_distance

    @staticmethod
    def _appearance_distance(
        track: ObjectTrack, detection: LocalizedDetection
    ) -> float | None:
        if (
            track.appearance_embedding is None
            or detection.appearance_embedding is None
            or not track.appearance_provider_id
            or track.appearance_provider_id != detection.appearance_provider_id
        ):
            return None
        if track.appearance_embedding.shape != detection.appearance_embedding.shape:
            return None
        distance = 1.0 - float(
            np.dot(track.appearance_embedding, detection.appearance_embedding)
        )
        return float(np.clip(distance, 0.0, 2.0)) if np.isfinite(distance) else None

    def _update_track(
        self, track: ObjectTrack, detection: LocalizedDetection
    ) -> None:
        now_sec = detection.stamp_ns / 1_000_000_000.0
        measurement = np.asarray(detection.position, dtype=np.float64)
        measurement_covariance = detection.position_covariance
        process_covariance = np.eye(3) * self.config.process_noise_stddev_m**2
        predicted_covariance = track.position_covariance + process_covariance
        innovation_covariance = predicted_covariance + measurement_covariance
        kalman_gain = np.linalg.solve(
            innovation_covariance.T, predicted_covariance.T
        ).T
        innovation = measurement - track.position
        track.position = track.position + kalman_gain @ innovation
        identity = np.eye(3, dtype=np.float64)
        residual = identity - kalman_gain
        covariance = (
            residual @ predicted_covariance @ residual.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        track.position_covariance = (covariance + covariance.T) / 2.0
        track.observation_count += 1
        track.first_seen_sec = min(track.first_seen_sec, now_sec)
        track.last_seen_sec = now_sec
        track.stale_since_sec = None
        track.model_id = self.model_id
        self._update_appearance(track, detection)
        self._add_class_observation(track, detection, now_sec)
        self._update_state(track, now_sec)

    def _new_track(self, detection: LocalizedDetection) -> ObjectTrack:
        now_sec = detection.stamp_ns / 1_000_000_000.0
        self._next_object_id += 1
        track = ObjectTrack(
            object_id=f"object_{self._next_object_id:03d}",
            position=np.asarray(detection.position, dtype=np.float64),
            position_covariance=detection.position_covariance.copy(),
            observation_count=1,
            first_seen_sec=now_sec,
            last_seen_sec=now_sec,
            model_id=self.model_id,
            appearance_embedding=_normalized_embedding(detection.appearance_embedding),
            appearance_provider_id=detection.appearance_provider_id,
        )
        self._add_class_observation(track, detection, now_sec)
        self._update_state(track, now_sec)
        return track

    def _update_appearance(
        self, track: ObjectTrack, detection: LocalizedDetection) -> None:
        embedding = _normalized_embedding(detection.appearance_embedding)
        if embedding is None or not detection.appearance_provider_id:
            return
        if (
            track.appearance_embedding is None
            or track.appearance_provider_id != detection.appearance_provider_id
            or track.appearance_embedding.shape != embedding.shape
        ):
            track.appearance_embedding = embedding
            track.appearance_provider_id = detection.appearance_provider_id
            return
        decay = self.config.appearance_embedding_decay
        track.appearance_embedding = _normalized_embedding(
            decay * track.appearance_embedding + (1.0 - decay) * embedding
        )

    def _add_class_observation(
        self,
        track: ObjectTrack,
        detection: LocalizedDetection,
        now_sec: float,
    ) -> None:
        normalized = _normalize_distribution(detection.class_likelihoods)
        range_m = max(float(detection.range_m), 1e-6)
        evidence_weight = min(
            1.0,
            max(
                self.config.distance_evidence_min_weight,
                self.config.distance_evidence_reference_m / range_m,
            ),
        )
        evidence_weight *= min(1.0, max(0.0, detection.class_evidence_strength))
        for label in list(track.class_scores):
            track.class_scores[label] *= self.config.evidence_decay
        for label, probability in normalized.items():
            track.class_scores[label] = (
                track.class_scores.get(label, 0.0) + probability * evidence_weight
            )
        track.class_scores.setdefault(OTHER_CLASS, 0.0)
        self._prune_class_scores(track)
        track.recent_observations.append((now_sec, normalized))

    def _prune_class_scores(self, track: ObjectTrack) -> None:
        explicit = [label for label in track.class_scores if label != OTHER_CLASS]
        while len(explicit) > self.config.max_explicit_classes:
            removed = min(explicit, key=lambda label: track.class_scores[label])
            track.class_scores[OTHER_CLASS] += track.class_scores.pop(removed)
            explicit.remove(removed)

    def _update_state(self, track: ObjectTrack, now_sec: float) -> None:
        cutoff = now_sec - self.config.confirmation_window_sec
        recent_count = sum(
            stamp >= cutoff for stamp, _ in track.recent_observations
        )
        if recent_count < self.config.confirmation_hits and not track.confirmation_locked:
            track.state = "candidate"
        elif (
            track.class_name == UNKNOWN_CLASS
            or track.class_probability < self.config.class_confirmation_probability
        ):
            track.state = "class_ambiguous"
        else:
            track.state = "confirmed"
            track.confirmation_locked = True

    def _validate_detection(self, detection: LocalizedDetection) -> None:
        position = np.asarray(detection.position, dtype=np.float64)
        covariance = np.asarray(detection.position_covariance, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Object position must contain three finite values")
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("Object covariance must be a finite 3x3 matrix")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("Object covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
            raise ValueError("Object covariance must be positive definite")
        _normalize_distribution(detection.class_likelihoods)
        embedding = _normalized_embedding(detection.appearance_embedding)
        if detection.appearance_embedding is not None and embedding is None:
            raise ValueError("Appearance embedding must be finite and non-zero")
        if embedding is not None and not detection.appearance_provider_id:
            raise ValueError("Appearance embedding requires a provider ID")
        if not np.isfinite(detection.range_m) or detection.range_m <= 0.0:
            raise ValueError("Detection range must be finite and positive")

    def _load_legacy_coordinates(self, payload: dict[str, object]) -> None:
        now_sec = datetime.now(tz=timezone.utc).timestamp()
        for raw_label, coordinates in payload.items():
            if str(raw_label).endswith("_approximate") or not isinstance(
                coordinates, dict
            ):
                continue
            try:
                position = np.array(
                    [coordinates["x"], coordinates["y"], coordinates["z"]],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite(position)):
                continue
            label = _normalize_label(raw_label)
            self._next_object_id += 1
            track = ObjectTrack(
                object_id=f"object_{self._next_object_id:03d}",
                position=position,
                position_covariance=np.eye(3) * 0.05**2,
                observation_count=self.config.confirmation_hits,
                first_seen_sec=now_sec,
                last_seen_sec=now_sec,
                model_id="legacy_json",
                class_scores={label: float(self.config.confirmation_hits)},
                state="confirmed",
                confirmation_locked=True,
            )
            self._tracks[track.object_id] = track

    def _load_versioned_item(self, item: object, schema_version: int) -> None:
        if not isinstance(item, dict):
            return
        try:
            position_data = item["position"]
            position = np.array(
                [position_data["x"], position_data["y"], position_data["z"]],
                dtype=np.float64,
            )
            object_id = str(item["id"])
            first_seen = _parse_time(str(item["first_seen"]))
            last_seen = _parse_time(str(item["last_seen"]))
            covariance = self._load_covariance(item, schema_version)
            class_scores = self._load_class_scores(item, schema_version)
        except (KeyError, TypeError, ValueError):
            return
        if not np.all(np.isfinite(position)):
            return
        try:
            appearance_embedding = self._load_appearance_embedding(item, schema_version)
        except (TypeError, ValueError):
            appearance_embedding = None
        track = ObjectTrack(
            object_id=object_id,
            position=position,
            position_covariance=covariance,
            observation_count=int(item.get("observation_count", 1)),
            first_seen_sec=first_seen,
            last_seen_sec=last_seen,
            model_id=str(item.get("model_id", "unknown")),
            appearance_embedding=appearance_embedding,
            appearance_provider_id=(
                str(item["appearance_provider_id"])
                if schema_version >= 3 and item.get("appearance_provider_id")
                else None
            ),
            class_scores=class_scores,
            state=str(item.get("state", "stale")),
            confirmation_locked=bool(
                item.get("confirmation_locked", item.get("state") == "confirmed")
            ),
            stale_since_sec=(
                _parse_time(str(item["stale_since"]))
                if item.get("stale_since")
                else None
            ),
        )
        self._prune_class_scores(track)
        self._tracks[track.object_id] = track
        self._record_loaded_id(track.object_id)

    @staticmethod
    def _load_covariance(item: dict[str, object], schema_version: int) -> np.ndarray:
        if schema_version >= 2:
            covariance = np.asarray(item["position_covariance"], dtype=np.float64)
            if covariance.size != 9:
                raise ValueError("Stored covariance must contain nine values")
            covariance = covariance.reshape(3, 3)
        else:
            stddev = float(item.get("position_stddev_m", 0.05))
            covariance = np.eye(3) * stddev**2
        if not np.all(np.isfinite(covariance)):
            raise ValueError("Stored covariance must be finite")
        return (covariance + covariance.T) / 2.0

    @staticmethod
    def _load_class_scores(
        item: dict[str, object], schema_version: int
    ) -> dict[str, float]:
        if schema_version >= 2:
            distribution = _normalize_distribution(
                dict(item.get("class_distribution", {}))
            )
            weight = max(1.0, float(item.get("class_evidence_weight", 1.0)))
            return {label: probability * weight for label, probability in distribution.items()}
        evidence = item.get("evidence", {})
        scores: dict[str, float] = {}
        if isinstance(evidence, dict):
            for label, values in evidence.items():
                if isinstance(values, dict):
                    score = float(values.get("confidence_sum", 0.0))
                    if score > 0.0:
                        scores[_normalize_label(label)] = score
        if not scores:
            label = _normalize_label(str(item.get("class_name", UNKNOWN_CLASS)))
            if label == UNKNOWN_CLASS:
                label = OTHER_CLASS
            scores[label] = 1.0
        scores.setdefault(OTHER_CLASS, 0.0)
        return scores

    @staticmethod
    def _load_appearance_embedding(
        item: dict[str, object], schema_version: int
    ) -> np.ndarray | None:
        if schema_version < 3 or "appearance_embedding" not in item:
            return None
        raw_embedding = item.get("appearance_embedding")
        if not isinstance(raw_embedding, list):
            return None
        return _normalized_embedding(np.asarray(raw_embedding, dtype=np.float64))

    def _record_loaded_id(self, object_id: str) -> None:
        if not object_id.startswith("object_"):
            return
        try:
            suffix = int(object_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return
        self._next_object_id = max(self._next_object_id, suffix)

    @staticmethod
    def _serialize_track(track: ObjectTrack) -> dict[str, object]:
        distribution = track.class_distribution
        return {
            "id": track.object_id,
            "class_name": track.class_name,
            "class_probability": round(float(track.class_probability), 6),
            "class_distribution": {
                label: round(float(probability), 6)
                for label, probability in sorted(distribution.items())
            },
            "class_evidence_weight": round(sum(track.class_scores.values()), 6),
            "position": {
                "x": round(float(track.position[0]), 6),
                "y": round(float(track.position[1]), 6),
                "z": round(float(track.position[2]), 6),
            },
            "position_covariance": [
                round(float(value), 10)
                for value in track.position_covariance.reshape(-1)
            ],
            "position_stddev_m": round(float(track.position_stddev_m), 6),
            "confidence": round(float(track.class_probability), 6),
            "state": track.state,
            "confirmation_locked": track.confirmation_locked,
            "observation_count": track.observation_count,
            "first_seen": _iso_time(track.first_seen_sec),
            "last_seen": _iso_time(track.last_seen_sec),
            **(
                {"stale_since": _iso_time(track.stale_since_sec)}
                if track.stale_since_sec is not None
                else {}
            ),
            "model_id": track.model_id,
            **(
                {
                    "appearance_provider_id": track.appearance_provider_id,
                    "appearance_embedding": [
                        round(float(value), 8)
                        for value in track.appearance_embedding.tolist()
                    ],
                }
                if track.appearance_embedding is not None
                and track.appearance_provider_id
                else {}
            ),
        }


def _normalize_label(label: str) -> str:
    normalized = str(label).strip().lower().replace(" ", "_")
    return normalized or UNKNOWN_CLASS


def _normalized_embedding(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    embedding = np.asarray(value, dtype=np.float64).reshape(-1)
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        return None
    norm = float(np.linalg.norm(embedding))
    if norm <= 0.0:
        return None
    return embedding / norm


def _finite_min(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else None


def _normalize_distribution(values: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_label, raw_probability in values.items():
        label = _normalize_label(raw_label)
        if label == UNKNOWN_CLASS:
            label = OTHER_CLASS
        probability = float(raw_probability)
        if not np.isfinite(probability) or probability < 0.0:
            raise ValueError("Class probabilities must be finite and non-negative")
        normalized[label] = normalized.get(label, 0.0) + probability
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("Class distribution must contain positive probability mass")
    normalized = {label: value / total for label, value in normalized.items()}
    normalized.setdefault(OTHER_CLASS, 0.0)
    return normalized


def _iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _atomic_json_write(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
