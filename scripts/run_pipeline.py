#!/usr/bin/env python3
"""Run the complete prototype."""

from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proximie_pipeline.pipeline import (
    evaluate,
    make_dataset,
    set_seed,
    train_model,
)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    set_seed(cfg["seed"])

    data_cfg = cfg
    train = make_dataset(
        cfg["train_sequences"],
        seed=cfg["seed"],
        num_sensors=cfg["num_sensors"],
        feature_dim=cfg["feature_dim_per_sensor"],
        sequence_length=cfg["sequence_length"],
    )
    val = make_dataset(
        cfg["validation_sequences"],
        seed=cfg["seed"] + 10,
        num_sensors=cfg["num_sensors"],
        feature_dim=cfg["feature_dim_per_sensor"],
        sequence_length=cfg["sequence_length"],
    )

    model = train_model(
        train,
        num_sensors=cfg["num_sensors"],
        feature_dim=cfg["feature_dim_per_sensor"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["num_classes"],
        dropout=cfg["model"]["dropout"],
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
    )

    result = evaluate(
        model,
        val,
        fps=cfg["evaluation"]["fps"],
        num_classes=cfg["num_classes"],
        median_kernel=cfg["postprocess"]["median_kernel"],
        min_segment_frames=cfg["postprocess"]["min_segment_frames"],
        merge_gap_frames=cfg["postprocess"]["merge_gap_frames"],
    )

    print("\n=== MODEL METRICS ===")
    for key, value in result.model_metrics.items():
        print(f"{key:45s} {value:.4f}")

    print("\n=== PRODUCT / SEGMENTATION METRICS ===")
    for key, value in result.product_metrics.items():
        print(f"{key:45s} {value:.4f}")


if __name__ == "__main__":
    main()
