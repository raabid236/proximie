"""Frame-to-segment temporal post-processing."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Segment:
    label: int
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


def median_filter_labels(labels: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return labels.copy()
    if kernel % 2 == 0:
        raise ValueError("kernel must be odd")
    pad = kernel // 2
    padded = np.pad(labels, (pad, pad), mode="edge")
    out = np.empty_like(labels)
    for i in range(len(labels)):
        out[i] = np.median(padded[i:i + kernel])
    return out.astype(labels.dtype)


def labels_to_segments(labels: Sequence[int]) -> list[Segment]:
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    segments: list[Segment] = []
    start = 0
    current = int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) != current:
            segments.append(Segment(current, start, i))
            start = i
            current = int(labels[i])
    segments.append(Segment(current, start, len(labels)))
    return segments


def clean_segments(
    labels: np.ndarray,
    *,
    median_kernel: int = 7,
    min_segment_frames: int = 12,
    merge_gap_frames: int = 4,
) -> list[Segment]:
    """Smooth labels, remove short islands, and merge tiny same-class gaps."""
    smoothed = median_filter_labels(labels, median_kernel)
    current = smoothed.copy()

    # Iteratively replace short islands with the longer neighboring class.
    for _ in range(len(current)):
        segments = labels_to_segments(current)
        short = next(
            (s for s in segments if s.duration < min_segment_frames),
            None,
        )
        if short is None or len(segments) == 1:
            break

        idx = segments.index(short)
        if idx == 0:
            target = segments[1].label
        elif idx == len(segments) - 1:
            target = segments[-2].label
        else:
            left, right = segments[idx - 1], segments[idx + 1]
            target = left.label if left.duration >= right.duration else right.label

        current[short.start:short.end] = target

    # Merge same-class segments separated by a short gap.
    segments = labels_to_segments(current)
    merged: list[Segment] = []
    for seg in segments:
        if (
            merged
            and seg.label == merged[-1].label
            and seg.start - merged[-1].end <= merge_gap_frames
        ):
            prev = merged.pop()
            merged.append(Segment(prev.label, prev.start, seg.end))
        else:
            merged.append(seg)
    return merged
