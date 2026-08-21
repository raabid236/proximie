#!/usr/bin/env python3
"""Mock targeted error analysis for the two challenge failure modes."""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proximie_pipeline.data import (
    EMPTY,
    PATIENT_PRESENT,
    OPERATION,
    POST_OPERATION,
)
from proximie_pipeline.metrics import frame_accuracy, per_class_f1
from proximie_pipeline.postprocess import clean_segments


def make_noisy_validation(seed: int = 7, n: int = 900):
    rng = np.random.default_rng(seed)
    y = np.full(n, EMPTY, dtype=np.int64)
    y[140:500] = PATIENT_PRESENT
    y[500:780] = OPERATION
    y[780:850] = POST_OPERATION

    p = y.copy()

    # Patient Present: fragmented false positives before surgery.
    for start in [35, 62, 91, 112]:
        p[start : start + 8] = PATIENT_PRESENT

    # Operation: boundary jitter and occlusion-driven dropout.
    p[485:505] = PATIENT_PRESENT
    p[520:535] = EMPTY
    p[610:625] = EMPTY
    p[715:730] = PATIENT_PRESENT
    p[770:790] = EMPTY

    # A few random background false positives.
    for idx in rng.choice(np.arange(0, 130), size=8, replace=False):
        p[idx] = PATIENT_PRESENT

    return y, p


def main() -> None:
    y, p = make_noisy_validation()

    raw_f1 = per_class_f1(y, p, 4)
    cleaned = clean_segments(
        p,
        median_kernel=7,
        min_segment_frames=12,
        merge_gap_frames=4,
    )
    pc = np.empty_like(p)
    for seg in cleaned:
        pc[seg.start:seg.end] = seg.label
    clean_f1 = per_class_f1(y, pc, 4)

    print("=== MOCK ERROR ANALYSIS ===")
    print(f"Raw frame accuracy:     {frame_accuracy(y, p):.3f}")
    print(f"Cleaned frame accuracy: {frame_accuracy(y, pc):.3f}")

    print("\nPatient Present")
    print("----------------")
    print(f"Raw F1:                 {raw_f1[PATIENT_PRESENT]:.3f}")
    print(f"Post-processed F1:      {clean_f1[PATIENT_PRESENT]:.3f}")
    print("Observed failure: short false-positive islands before surgery.")
    print("Likely mitigation: class-specific minimum duration + context gate.")

    print("\nOperation")
    print("---------")
    print(f"Raw F1:                 {raw_f1[OPERATION]:.3f}")
    print(f"Post-processed F1:      {clean_f1[OPERATION]:.3f}")
    print("Observed failure: occlusion-driven holes and boundary jitter.")
    print("Likely mitigation: quality-aware multi-view fusion + boundary head.")


if __name__ == "__main__":
    main()
