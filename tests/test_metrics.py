import numpy as np

from proximie_pipeline.metrics import (
    frame_accuracy,
    interval_iou,
    per_class_f1,
)
from proximie_pipeline.postprocess import Segment


def test_frame_accuracy():
    y = np.array([0, 1, 1, 2])
    p = np.array([0, 1, 2, 2])
    assert frame_accuracy(y, p) == 0.75


def test_per_class_f1_is_perfect():
    y = np.array([0, 1, 2, 1])
    assert per_class_f1(y, y, 3)[1] == 1.0


def test_interval_iou():
    a = Segment(1, 10, 20)
    b = Segment(1, 15, 25)
    assert interval_iou(a, b) == 5 / 15
