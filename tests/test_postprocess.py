import numpy as np

from proximie_pipeline.postprocess import clean_segments, labels_to_segments


def test_labels_to_segments():
    labels = np.array([0, 0, 1, 1, 2, 2, 2])
    segments = labels_to_segments(labels)
    assert [(s.label, s.start, s.end) for s in segments] == [
        (0, 0, 2),
        (1, 2, 4),
        (2, 4, 7),
    ]


def test_postprocess_removes_short_island():
    labels = np.array([1] * 20 + [2] * 3 + [1] * 20)
    segments = clean_segments(
        labels, median_kernel=1, min_segment_frames=5, merge_gap_frames=4
    )
    assert len(segments) == 1
    assert segments[0].label == 1
