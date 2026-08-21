"""End-to-end training/evaluation orchestration."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import MockFeatureStream, SequenceExample
from .metrics import (
    frame_accuracy,
    macro_f1,
    per_class_f1,
    segmentation_product_metrics,
)
from .model import TemporalGRU, fuse_sensor_features
from .postprocess import clean_segments


@dataclass
class PipelineResult:
    model_metrics: dict[str, float]
    product_metrics: dict[str, float]
    predictions: np.ndarray
    labels: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(
    count: int,
    *,
    seed: int,
    num_sensors: int,
    feature_dim: int,
    sequence_length: int,
) -> list[SequenceExample]:
    data: list[SequenceExample] = []
    for i in range(count):
        stream = MockFeatureStream(
            num_sensors=num_sensors,
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            seed=seed + i,
        )
        data.extend(list(stream))
    return data


def stack_batch(examples: list[SequenceExample]):
    x = torch.tensor(np.stack([e.features for e in examples]), dtype=torch.float32)
    q = torch.tensor(np.stack([e.sensor_quality for e in examples]), dtype=torch.float32)
    y = torch.tensor(np.stack([e.labels for e in examples]), dtype=torch.long)
    return x, q, y


def train_model(
    train: list[SequenceExample],
    *,
    num_sensors: int,
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_classes: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> TemporalGRU:
    model = TemporalGRU(
        input_dim=feature_dim + 2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        order = np.random.permutation(len(train))
        for start in range(0, len(train), batch_size):
            batch = [train[i] for i in order[start : start + batch_size]]
            x, q, y = stack_batch(batch)
            fused = fuse_sensor_features(x, q)
            logits = model(fused)
            loss = criterion(logits.reshape(-1, num_classes), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict(model: TemporalGRU, examples: list[SequenceExample]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_pred, all_true = [], []
    for example in examples:
        x, q, y = stack_batch([example])
        logits = model(fuse_sensor_features(x, q))
        all_pred.append(logits.argmax(dim=-1).cpu().numpy()[0])
        all_true.append(y.cpu().numpy()[0])
    return np.concatenate(all_true), np.concatenate(all_pred)


def evaluate(
    model: TemporalGRU,
    validation: list[SequenceExample],
    *,
    fps: float,
    num_classes: int,
    median_kernel: int,
    min_segment_frames: int,
    merge_gap_frames: int,
) -> PipelineResult:
    y_true, y_pred_raw = predict(model, validation)

    # Evaluate frame-level model output separately.
    model_metrics = {
        "frame_accuracy": frame_accuracy(y_true, y_pred_raw),
        "macro_f1": macro_f1(y_true, y_pred_raw, num_classes),
    }
    for c, value in per_class_f1(y_true, y_pred_raw, num_classes).items():
        model_metrics[f"f1_class_{c}"] = value

    # Product metrics use independently post-processed per-sequence timelines.
    product_accum = []
    offset = 0
    for example in validation:
        n = len(example.labels)
        pred = y_pred_raw[offset : offset + n]
        cleaned = clean_segments(
            pred,
            median_kernel=median_kernel,
            min_segment_frames=min_segment_frames,
            merge_gap_frames=merge_gap_frames,
        )
        cleaned_labels = np.empty_like(pred)
        for seg in cleaned:
            cleaned_labels[seg.start:seg.end] = seg.label
        product_accum.append(
            segmentation_product_metrics(
                example.labels,
                cleaned_labels,
                fps=fps,
                num_classes=num_classes,
            )
        )
        offset += n

    keys = product_accum[0].keys()
    product_metrics = {
        key: float(np.nanmean([m[key] for m in product_accum]))
        for key in keys
    }
    return PipelineResult(model_metrics, product_metrics, y_pred_raw, y_true)
