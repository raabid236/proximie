"""Model and product-oriented segmentation metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .postprocess import Segment, labels_to_segments


def frame_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores[c] = float(
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    return scores


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    return float(np.mean(list(per_class_f1(y_true, y_pred, num_classes).values())))


def interval_iou(a: Segment, b: Segment) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union else 1.0


def greedy_segment_matches(
    true_segments: Iterable[Segment],
    pred_segments: Iterable[Segment],
    iou_threshold: float = 0.5,
) -> list[tuple[Segment, Segment]]:
    true_by_class = defaultdict(list)
    pred_by_class = defaultdict(list)
    for s in true_segments:
        true_by_class[s.label].append(s)
    for s in pred_segments:
        pred_by_class[s.label].append(s)

    matches: list[tuple[Segment, Segment]] = []
    for label, trues in true_by_class.items():
        remaining = set(range(len(pred_by_class[label])))
        for t in trues:
            best = None
            best_iou = 0.0
            for j in remaining:
                p = pred_by_class[label][j]
                score = interval_iou(t, p)
                if score > best_iou:
                    best_iou = score
                    best = j
            if best is not None and best_iou >= iou_threshold:
                matches.append((t, pred_by_class[label][best]))
                remaining.remove(best)
    return matches


def boundary_mae(
    true_segments: Iterable[Segment],
    pred_segments: Iterable[Segment],
    iou_threshold: float = 0.1,
) -> float:
    matches = greedy_segment_matches(true_segments, pred_segments, iou_threshold)
    if not matches:
        return float("nan")
    errors = [
        (abs(t.start - p.start) + abs(t.end - p.end)) / 2
        for t, p in matches
    ]
    return float(np.mean(errors))


def segmentation_product_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    fps: float,
    num_classes: int,
) -> dict[str, float]:
    true_segments = labels_to_segments(y_true)
    pred_segments = labels_to_segments(y_pred)
    matches = greedy_segment_matches(true_segments, pred_segments, 0.5)

    per_class_iou = defaultdict(list)
    for t, p in matches:
        per_class_iou[t.label].append(interval_iou(t, p))

    all_ious = [v for values in per_class_iou.values() for v in values]
    boundary = boundary_mae(true_segments, pred_segments, 0.1)

    # Fragmentation: excess predicted segments for a class relative to truth.
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    for s in true_segments:
        true_counts[s.label] += 1
    for s in pred_segments:
        pred_counts[s.label] += 1
    frag_values = []
    for c in range(num_classes):
        if true_counts[c]:
            frag_values.append(max(0, pred_counts[c] - true_counts[c]) / true_counts[c])
    fragmentation = float(np.mean(frag_values)) if frag_values else 0.0

    pp_false_frames = np.sum(
        (y_true != 1) & (y_pred == 1)
    )
    pp_false_duration_s = float(pp_false_frames / fps)

    return {
        "segment_iou_mean": float(np.mean(all_ious)) if all_ious else 0.0,
        "boundary_mae_seconds": (
            boundary / fps if np.isfinite(boundary) else float("nan")
        ),
        "fragmentation_rate": fragmentation,
        "patient_present_false_positive_seconds": pp_false_duration_s,
        "true_segment_count": float(len(true_segments)),
        "pred_segment_count": float(len(pred_segments)),
    }
