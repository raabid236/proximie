"""Mock multi-sensor feature ingestion.

The production interface is intentionally represented as [T, S, D]:
T = time steps, S = sensors, D = visual embedding dimensions.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

EMPTY = 0
PATIENT_PRESENT = 1
OPERATION = 2
POST_OPERATION = 3


@dataclass(frozen=True)
class SequenceExample:
    features: np.ndarray          # [T, S, D]
    labels: np.ndarray            # [T]
    sensor_quality: np.ndarray    # [T, S]


class MockFeatureStream:
    """Deterministic generator of synthetic OR feature streams."""

    def __init__(
        self,
        *,
        num_sensors: int = 3,
        feature_dim: int = 16,
        sequence_length: int = 600,
        seed: int = 42,
    ) -> None:
        self.num_sensors = num_sensors
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)

    def _phase_schedule(self) -> np.ndarray:
        """Create a plausible workflow timeline."""
        t = self.sequence_length
        labels = np.full(t, EMPTY, dtype=np.int64)

        # Keep the long static Patient Present phase intentionally long.
        pp_start = int(0.18 * t)
        op_start = int(0.52 * t)
        op_end = int(0.82 * t)
        post_end = int(0.94 * t)

        labels[pp_start:op_start] = PATIENT_PRESENT
        labels[op_start:op_end] = OPERATION
        labels[op_end:post_end] = POST_OPERATION
        return labels

    def __iter__(self) -> Iterator[SequenceExample]:
        labels = self._phase_schedule()

        # Class prototypes in embedding space.
        prototypes = self.rng.normal(
            0.0, 0.7, size=(4, self.feature_dim)
        )

        for _ in range(1):
            t = self.sequence_length
            base = prototypes[labels].copy()

            # Slow environmental drift = background movement / lighting changes.
            drift = np.cumsum(
                self.rng.normal(0.0, 0.015, size=(t, self.feature_dim)),
                axis=0,
            )
            drift -= drift.mean(axis=0, keepdims=True)

            features = np.empty(
                (t, self.num_sensors, self.feature_dim), dtype=np.float32
            )
            quality = np.empty((t, self.num_sensors), dtype=np.float32)

            for sensor in range(self.num_sensors):
                sensor_bias = self.rng.normal(
                    0.0, 0.12, size=(self.feature_dim,)
                )
                noise_scale = np.where(
                    labels == OPERATION, 0.65, 0.25
                )
                noise = self.rng.normal(
                    0.0,
                    noise_scale[:, None],
                    size=(t, self.feature_dim),
                )
                q = np.ones(t, dtype=np.float32)

                # Occlusion is deliberately concentrated in Operation.
                op_mask = labels == OPERATION
                occluded = op_mask & (
                    self.rng.random(t) < (0.18 + 0.08 * sensor)
                )
                q[occluded] = 0.15

                sensor_features = base + drift + sensor_bias + noise
                # A low-quality view loses some class information.
                sensor_features[occluded] *= 0.15

                features[:, sensor, :] = sensor_features
                quality[:, sensor] = q

            yield SequenceExample(
                features=features,
                labels=labels,
                sensor_quality=quality,
            )
